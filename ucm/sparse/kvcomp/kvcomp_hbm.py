from typing import Any, Dict, List, Optional, Union

import torch
from vllm import _custom_ops as ops
from vllm.attention.ops.flashmla import get_mla_metadata
from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.request import Request, RequestStatus

from ucm.logger import init_logger
from ucm.sparse.base import (
    INVALID_SLOT,
    UcmSparseBase,
    UcmSparseRole,
)
from ucm.sparse.kvcomp.hamming_topk import cuda_hamming_topk, fake_hamming_topk
from ucm.sparse.kvcomp.hash_encoder import HashEncoder, triton_hash_code
from vllm.attention.utils.fa_utils import reshape_and_cache_flash
from ucm.sparse.kvcomp.kvcomp_config import KvCompConfig
from ucm.utils import Config

logger = init_logger(__name__)

ReqType = Union[str, int]

def require_mla_mode(func):
    def wrapper(self, *args, **kwargs):
        if not self.is_mla:
            raise RuntimeError(
                f"Method {func.__name__} can only be called in MLA model mode (is_deepseek_mla=False)"
            )
        return func(self, *args, **kwargs)
    return wrapper

def require_gqa_mode(func):
    def wrapper(self, *args, **kwargs):
        if self.is_mla:
            raise RuntimeError(
                f"Method {func.__name__} can only be called in GQA model mode (is_deepseek_mla=True)"
            )
        return func(self, *args, **kwargs)
    return wrapper

class KvCompOnDevice(UcmSparseBase):
    # handle batch
    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        super().__init__(vllm_config, role)

        self.rank = vllm_config.parallel_config.rank
        self.is_mla = vllm_config.model_config.is_deepseek_mla
        self.device = torch.device(f"cuda:{self.rank}")
        self.num_q_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_key_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.block_size = vllm_config.cache_config.block_size

        self.kvcompOnDevice_cfg = (
            Config(vllm_config.kv_transfer_config)
            .get_config()
            .get("ucm_sparse_config")
            .get("KvCompOnDevice")
        )

        kvcompOnDevice_config_path = self.kvcompOnDevice_cfg["kvcompOnDevice_config_path"]
        self.kvcompOnDevice_config = KvCompConfig.from_json(kvcompOnDevice_config_path)
        logger.info(f"read kvcomp config file : {kvcompOnDevice_config_path} ")
        self.hash_topk = self.kvcompOnDevice_config.vllm_hash_attention_topk
        self.hash_rollback_layers = self.kvcompOnDevice_config.vllm_hash_attention_rollback_layers
        self.hash_skip_layers = self.kvcompOnDevice_config.vllm_hash_attention_skip_layers

        if role == UcmSparseRole.WORKER:
            device_properties = torch.cuda.get_device_properties(self.device)
            num_sms = device_properties.multi_processor_count

            if not vllm_config.model_config.enforce_eager:
                self.cg_buf_topk_tile_scheduler_metadata = torch.zeros(
                    (num_sms, 8),
                    device=self.device,
                    dtype=torch.int32,
                )
                self.cg_buf_topk_num_splits = torch.empty(
                    (vllm_config.scheduler_config.max_num_seqs + 1),
                    device=self.device,
                    dtype=torch.int32,
                )

            self.origin_block_table = None
            self.origin_seq_lens = None
            self.origin_tile_scheduler_metadata = None
            self.origin_num_splits = None

            self.ori_seq_lens_decode = None
            self.ori_block_table_decode = None

            if self.is_mla:
                logger.info("KvCompOnDevice initialized with MLA model config")
                self.hash_reduction_head_num = self.kvcompOnDevice_config.vllm_hash_attention_reduction_head_num
                self.kv_lora_rank = getattr(
                    vllm_config.model_config.hf_text_config, "kv_lora_rank", None
                )
                self.qk_rope_head_dim = getattr(
                    vllm_config.model_config.hf_text_config, "qk_rope_head_dim", None
                )
                self.hash_encoder_nope = HashEncoder(
                    input_dim=self.kv_lora_rank,
                    hash_bits=self.kv_lora_rank,
                    dtype=vllm_config.model_config.dtype,
                    device=self.device,
                )

                self.hash_encoder_rope = HashEncoder(
                    input_dim=self.qk_rope_head_dim,
                    hash_bits=self.qk_rope_head_dim,
                    dtype=vllm_config.model_config.dtype,
                    device=self.device,
                )
            else:
                logger.info("KvCompOnDevice initialized with non-MLA model config")
                self.head_dim = vllm_config.model_config.get_head_size()
                self.hash_encoder = HashEncoder(
                    input_dim=self.head_dim,
                    hash_bits=self.head_dim,
                    dtype=vllm_config.model_config.dtype,
                    device=self.device,
                )
            self._k_scale = torch.tensor(1.0, dtype=torch.float32)
            self._v_scale = torch.tensor(1.0, dtype=torch.float32)

    @require_mla_mode
    def hash_code_mla(self, nope, rope, reduction_head_num=1):
        if reduction_head_num > 1:
            nope = nope.view(
                nope.shape[0],
                reduction_head_num,
                nope.shape[1] // reduction_head_num,
                nope.shape[2],
            ).mean(dim=1)
            rope = rope.view(
                rope.shape[0],
                reduction_head_num,
                rope.shape[1] // reduction_head_num,
                rope.shape[2],
            ).mean(dim=1)

        hash_nope = self.hash_encoder_nope.compute_hash(nope)
        hash_rope = self.hash_encoder_rope.compute_hash(rope)
        return hash_nope.view(torch.bfloat16), hash_rope.view(torch.bfloat16)

    @require_gqa_mode
    def hash_code_gqa(self, query):
        if self.num_q_heads > self.num_key_heads:
            query = query.view(
                query.shape[0],
                self.num_key_heads,
                self.num_q_heads // self.num_key_heads,
                query.shape[2],
            )
            query = query.mean(2)
        elif self.num_q_heads < self.num_key_heads:
            query = torch.repeat_interleave(query, self.num_key_heads // self.num_q_heads, 1)

        hash_query = self.hash_encoder.compute_hash(query)
        return hash_query.view(torch.bfloat16)

    
    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
        output: Optional[torch.Tensor] = None,
        phase: Optional[str] = None,
        k_hash: Optional[torch.Tensor] = None,
        v_hash: Optional[torch.Tensor] = None,
        decode_ql_nope: Optional[torch.Tensor] = None,
        decode_q_pe: Optional[torch.Tensor] = None,
    ):
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[layer_name]

        layer_id = int(layer_name.split(".")[2])
        # TODO: Should mark MTP layer as rollback layer
        is_rollback_layer = layer_id in self.hash_rollback_layers
        is_skip_hash_layer = (
            layer_id < len(self.hash_skip_layers)
            and self.hash_skip_layers[layer_id]
        )

        if (not is_rollback_layer
            and not is_skip_hash_layer
        ):
            if self.is_mla:
                k_c_normed_hash, k_pe_hash = self.hash_code_mla(key, value)
                ops.concat_and_cache_mla(
                    k_c_normed_hash,
                    k_pe_hash.squeeze(1),
                    k_hash,
                    attn_metadata.slot_mapping.flatten(),
                    kv_cache_dtype="auto",
                    scale=self._k_scale,
                )
            else:
                k_hash_compute = self.hash_encoder.compute_hash(key)
                k_hash_compute = k_hash_compute.view(torch.bfloat16)
                v_dummy_tensor = torch.zeros(
                    v_hash.shape,
                    dtype=v_hash.dtype,
                    device=v_hash.device,
                )
                reshape_and_cache_flash(
                    k_hash_compute,
                    v_dummy_tensor,
                    k_hash,
                    v_hash, #bypass 
                    attn_metadata.slot_mapping,
                    kv_cache_dtype="auto",
                    k_scale=self._k_scale,
                    v_scale=self._k_scale,
                )
        if self.is_mla:
            if phase == "decode":
                if not is_rollback_layer:
                    if is_skip_hash_layer:
                        assert attn_metadata.decode.topk_block_table is not None
                        block_table = attn_metadata.decode.topk_block_table
                    else:
                        q_hash = torch.cat(
                            self.hash_code_mla(
                                decode_ql_nope,
                                decode_q_pe,
                                reduction_head_num=self.hash_reduction_head_num,
                            ),
                            dim=-1,
                        )
                        topk_token = self.hash_topk
                        block_table = cuda_hamming_topk(
                            q_hash.unsqueeze(1),
                            k_hash.unsqueeze(1),
                            attn_metadata.decode.block_table,
                            attn_metadata.decode.seq_lens,
                            topk_token=topk_token,
                            sink_token=64,
                            recent_token=512,
                        )
                        attn_metadata.decode.topk_block_table = block_table

                    seq_lens = attn_metadata.decode.topk_seq_lens
                    tile_scheduler_metadata = (
                        attn_metadata.decode.topk_tile_scheduler_metadata
                    )
                    num_splits = attn_metadata.decode.topk_num_splits

                    self.origin_block_table = attn_metadata.decode.block_table
                    self.origin_seq_lens = attn_metadata.decode.seq_lens
                    self.origin_tile_scheduler_metadata = (
                        attn_metadata.decode.tile_scheduler_metadata
                    )
                    self.origin_num_splits = attn_metadata.decode.num_splits

                    attn_metadata.decode.block_table = block_table
                    attn_metadata.decode.seq_lens = seq_lens
                    attn_metadata.decode.tile_scheduler_metadata = tile_scheduler_metadata
                    attn_metadata.decode.num_splits = num_splits
        else:
            q_start = attn_metadata.query_start_loc
            q_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
            decode_mask = (q_lens == 1)
            
            if not is_rollback_layer:
                if is_skip_hash_layer:
                    assert attn_metadata.block_tables is not None
                    block_table = attn_metadata.block_tables
                else:
                    if decode_mask.any():
                        decode_req_ids = torch.nonzero(decode_mask, as_tuple=False).flatten()
                        decode_token_idx = q_start[:-1][decode_mask]
                        q_decode = query.index_select(0, decode_token_idx)
                        q_hash = self.hash_code_gqa(q_decode)
                        
                        topk_token = self.hash_topk
                        
                        block_table_decode = attn_metadata.block_tables.index_select(0, decode_req_ids)
                        seq_len_decode = seq_lens.index_select(0, decode_req_ids)

                        block_table_decode = cuda_hamming_topk(
                            q_hash.unsqueeze(1),
                            k_hash.unsqueeze(1),
                            block_table_decode,
                            seq_len_decode,
                            topk_token=topk_token,
                            sink_token=64,
                            recent_token=512,
                        )
                        attn_metadata.block_tables = block_table
                        new_block_tables = attn_metadata.block_tables.clone()
                        new_block_tables.index_copy_(0, decode_req_ids, block_table_decode)

                seq_lens = attn_metadata.seq_lens
                self.origin_attn_metadata = attn_metadata
                attn_metadata.block_table = block_table
                attn_metadata.seq_lens = seq_lens
            else:
                # FA使用的是原本的 block_table和seq_lens
                self.origin_block_table = block_table
                attn_metadata.seq_lens = self.origin_seq_lens
                    

        return query, key, value, output

    def attention_finished(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_output: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
        phase: Optional[str] = None,
    ) -> None:
        layer_id = int(layer_name.split(".")[2])
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[layer_name]
        is_rollback_layer = layer_id in self.hash_rollback_layers 
        if self.is_mla:
            if phase == "decode":
                # TODO: Should mark MTP layer as rollback layer
                if not is_rollback_layer:
                    attn_metadata.decode.block_table = self.origin_block_table
                    attn_metadata.decode.seq_lens = self.origin_seq_lens
                    attn_metadata.decode.tile_scheduler_metadata = (
                        self.origin_tile_scheduler_metadata
                    )
                    attn_metadata.decode.num_splits = self.origin_num_splits
        else: # 判断req decode阶段
            if not is_rollback_layer:
                attn_metadata.block_table = self.origin_attn_metadata.block_table
                attn_metadata.seq_lens = self.origin_attn_metadata.seq_lens
                

    def request_begin(self, request_id: ReqType, prompt_token_ids: List[int]):
        pass

    def request_finished_in_scheduler(self, request_id: Union[int, str]):
        """
        This is called inside "Scheduler->finish_requests" function.
        Generate the metadata required by UcmSparse instance at worker-side.
        """
        pass

    def estimate_num_slots_sparsed(self, request: Request) -> int:
        return INVALID_SLOT

    def initialize_kv_hash_cache_tensors(self, kv_caches, device):
        dtype = torch.bfloat16
        for layer_name, kv_cache in kv_caches.items():
            khash_cache_shape = list(kv_cache.shape)
            khash_cache_shape[-1] //= dtype.itemsize * 8
            khash_cache = torch.zeros(khash_cache_shape, dtype=dtype, device=device)
            if self.is_mla:
                kv_caches[layer_name] = (kv_cache, khash_cache)
            else:
                dummy_v_cache = torch.zeros(
                    (1, 1, 1),
                    dtype=kv_cache.dtype,
                    device=device,
                )
                kv_caches[layer_name] = (kv_cache, khash_cache, dummy_v_cache)
           

    def build_decode_hash(self, seq_lens):
        from ucm.sparse.kvcomp.hamming_topk import update_seq_lens

        topk_seq_lens = update_seq_lens(
            seq_lens,
            topk_token=self.hash_topk,
            block_size=self.block_size,
        )
        topk_tile_scheduler_metadata, topk_num_splits = get_mla_metadata(
            topk_seq_lens,
            self.num_q_heads,
            1,
        )
        return topk_seq_lens, topk_tile_scheduler_metadata, topk_num_splits
    
    def build_decode_attention_meta(self, query_start_loc, seq_lens, block_table):
        from ucm.sparse.kvcomp.hamming_topk import update_seq_lens
        q_lens = query_start_loc[1:] - query_start_loc[:-1]
        decode_mask = (q_lens == 1)
        if decode_mask.any():
            decode_seq_lens = seq_lens[decode_mask]
            self.ori_seq_lens_decode = seq_lens
            self.ori_block_table_decode = block_table
            topk_seq_lens = update_seq_lens(
                decode_seq_lens,
                topk_token=self.hash_topk,
                block_size=self.block_size,
            )
        return decode_mask, topk_seq_lens

    def maybe_init_cudagraph_buffers_for_topk(self, n, tile_scheduler_metadata):
        sm_parts = tile_scheduler_metadata.size(0)
        topk_tile_scheduler_metadata_view = (
            self.cg_buf_topk_tile_scheduler_metadata[:sm_parts]
        )
        topk_tile_scheduler_metadata_view.copy_(topk_tile_scheduler_metadata)
        topk_tile_scheduler_metadata = topk_tile_scheduler_metadata_view

        topk_num_splits_view = self.cg_buf_topk_num_splits[:n]
        topk_num_splits_view.copy_(topk_num_splits)
        self.cg_buf_topk_num_splits[n:].fill_(topk_num_splits[-1])
        topk_num_splits = topk_num_splits_view
        return topk_tile_scheduler_metadata, topk_num_splits
