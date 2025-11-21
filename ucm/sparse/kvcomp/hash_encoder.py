"""
The MIT License

Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import torch

if hasattr(torch, "npu") and torch.npu.is_available():
    import torch_npu

#from ucm.logger import init_logger

#logger = init_logger(__name__)

if hasattr(torch, "cuda") and torch.cuda.is_available():
    import triton
    import triton.language as tl

    @triton.jit
    def triton_matmul_kernel(x_flat, hash_weights, output, BLOCK_SIZE: tl.constexpr):
        """
        Triton kernel to perform matrix multiplication (x_flat @ hash_weights) and project values to uint8.
        Args:
            x_flat: Input tensor of shape [N, input_dim]
            hash_weights: Weight matrix of shape [input_dim, hash_bits]
            output: Output tensor to store the projected values as uint8
            BLOCK_SIZE: Block size for Triton kernel
        """
        # Get program id (block id) for row
        pid = tl.program_id(0)
        idx = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int32) # Thread index in the block

        # Perform matrix multiplication: x_flat @ hash_weights
        matmul_result = tl.dot(x_flat[idx, :], hash_weights)

        # Project the result to uint8 by comparing to 0 (matmul_result > 0)
        projected = (matmul_result > 0).to(tl.uint8)

        # Store the projected result into the output tensor
        tl.store(output + idx, projected)

    @triton.jit
    def triton_pack_hash_kernel(output, bit_masks, packed_codes, hash_numbers, BLOCK_SIZE: tl.constexpr):
        """
        Triton kernel to pack the binary values into uint8 hash codes.
        Args:
            output: Projected values as uint8 with shape [N, hash_bits]
            bit_masks: The bit masks used for packing the bits
            packed_codes: The packed uint8 hash codes
            N: Number of rows (samples)
            hash_numbers: Number of uint8 numbers (hash_bits // 8)
            BLOCK_SIZE: Block size for Triton kernel
        """
        # Get program id (block id) for row
        pid = tl.program_id(0)
        idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # Thread index in the block

        # Load the projected values for the current thread
        projected = tl.load(output + idx)  # Shape: [N, hash_bits]

        # [N, hash_numbers, 8]
        binary_codes = projected.view(-1, hash_numbers, 8)

        # Perform bitwise packing: binary_codes * bit_masks
        packed_codes_flat = tl.sum(binary_codes * bit_masks, axis=-1, dtype=tl.uint8)

        # Store the packed codes into the packed_codes tensor
        tl.store(packed_codes + idx, packed_codes_flat)

class HashEncoder:
    """
    HashEncoder converts a float tensor to a binary hash code tensor,
    and it packs every 8 bits into a uint8 number.
    """

    def __init__(
        self, input_dim: int, hash_bits: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        self.input_dim = input_dim

        if hash_bits % 8 != 0:
            raise ValueError("hash_bits must be a multiple of 8")

        self.hash_bits = hash_bits

        # number of uint8 numbers to store hash_bits bits
        self.hash_numbers = self.hash_bits // 8

        self.dtype = dtype
        self.device = device

        if self.device.type == "npu":
            if dtype not in [torch.float16, torch.float32, torch.float64]:
                logger.warning(
                    "NPU only supports float16, float32 and float64 for hash_weights"
                )
                logger.warning("automatically using  float16 for hash_weights now")
                self.dtype = torch.float16

        if self.device.type == "cuda" and dtype == torch.bfloat16:
            logger.warning("geqrf_cuda not implemented for BFloat16")
            logger.warning("automatically using  float32 for hash_weights now")
            self.dtype = torch.float32

        # Step 1: 随机高斯矩阵
        random_weights = torch.normal(
            mean=0,
            std=2,
            size=(self.input_dim, self.hash_bits),
            dtype=self.dtype,
            device=self.device,
        )
        # Step 2: QR分解
        Q, R = torch.linalg.qr(random_weights)

        # Step 3: 调整符号，保证Haar 分布
        d = torch.sign(torch.diag(R))
        self.hash_weights = Q * d

        if self.device.type == "cuda" or self.device.type == "cpu":
            self._init_bit_masks()

    def set_hash_weight(self, hash_weights: torch.Tensor) -> None:
        if hash_weights.shape != (self.input_dim, self.hash_bits):
            raise ValueError(
                f"hash_weights shape {hash_weights.shape} does not match required shape {(self.input_dim, self.hash_bits)}"
            )
        if hash_weights.dtype != self.dtype:
            raise ValueError(
                f"hash_weights dtype {hash_weights.dtype} does not match required dtype {self.dtype}"
            )
        if hash_weights.device != self.device:
            raise ValueError(
                f"hash_weights device {hash_weights.device} does not match required device {self.device}"
            )

        self.hash_weights.copy_(hash_weights)

    def _init_bit_masks(self) -> None:
        self.bit_masks = torch.pow(
            2, torch.arange(8, dtype=torch.uint8, device=self.device)
        )
        # shape (1, 1, 8)
        self.bit_masks = self.bit_masks.unsqueeze(0).unsqueeze(0)

    def compute_hash(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the hash code for input tensor x.
        Args:
            x: input tensor of shape (..., input_dim)
        Returns:
            A tensor of shape (..., hash_numbers=hash_bits // 8) representing the hash codes.
            Each element is a uint8 number representing 8 bits of the hash code.
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"x must be of shape (..., {self.input_dim}), but got {x.shape}"
            )
        if x.device != self.device:
            raise ValueError(
                f"x device {x.device} does not match required device {self.device}"
            )

        # original shape without the last dimension
        # e.g. x.shape=[s1,s2,s3,input_dim], orig_shape=[s1,s2,s3]
        orig_shape = x.shape[:-1]

        # [N, input_dim], e.g., N = s1*s2*s3
        x_flat = x.view(-1, self.input_dim)

        if x_flat.dtype != self.dtype:
            x_flat = x_flat.to(self.dtype)

        if self.device.type == "npu":
            # [N, hash_bits]
            xW = torch.matmul(x_flat, self.hash_weights)
            # [N * hash_bits]
            xW_flat = xW.view(-1)
            # [N*hash_numbers], where hash_numbers = hash_bits // 8
            packed_codes_flat = torch_npu.npu_sign_bits_pack(xW_flat, size=1)
        
        elif self.device.type == "cuda":
            N = x_flat.shape[0]
            BLOCK_SIZE = 32

            # Allocate output tensors
            projected_output = torch.empty((N, self.hash_bits), dtype=torch.uint8, device=self.device)
            packed_codes = torch.empty((N, self.hash_numbers), dtype=torch.uint8, device=self.device)

            # Launch Triton kernel for matrix multiplication and projection
            grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
            triton_matmul_kernel[grid](x_flat, self.hash_weights, projected_output, BLOCK_SIZE=BLOCK_SIZE)

            # Launch Triton kernel for packing binary values
            grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
            triton_pack_hash_kernel[grid](projected_output, self.bit_masks, packed_codes, self.hash_numbers, BLOCK_SIZE=BLOCK_SIZE)
            packed_codes_flat = packed_codes.view(-1)  # [N * hash_numbers]
            
        elif self.device.type == "cpu":
            # [N, hash_bits]
            xW = torch.matmul(x_flat, self.hash_weights)
            # [N, hash_bits]
            projected = (xW > 0).to(torch.uint8)
            # [N, hash_numbers, 8]
            binary_codes = projected.view(-1, self.hash_numbers, 8)

            # binary_codes * self.bit_masks [N, hash_numbers, 8] * [1, 1, 8] -> [N, hash_numbers, 8]
            # then sum along the last dimension to get [N, hash_numbers]
            packed_codes_flat = torch.sum(
                binary_codes * self.bit_masks, dim=-1, dtype=torch.uint8
            )  # [N, hash_numbers]
            packed_codes_flat = packed_codes_flat.view(-1)  # [N * hash_numbers]
        else:
            raise ValueError(f"Unsupported device type: {self.device.type}")

        # e.g., [s1, s2, s3, hash_numbers]
        out_shape = orig_shape + (self.hash_numbers,)
        packed_codes = packed_codes_flat.view(out_shape)

        return packed_codes

    def _unpack_hash(self, packed_codes: torch.Tensor) -> torch.Tensor:
        """
        Unpack the hash codes to +1 or -1 bits.
        Args:
            packed_codes: input tensor of shape (..., hash_numbers), dtype=torch.uint8
        Returns:
            A tensor of shape (..., hash_bits=hash_numbers*8) representing the unpacked bits.
            Each element is either -1 or 1.
        """
        if packed_codes.shape[-1] != self.hash_numbers:
            raise ValueError(
                f"packed_codes must be of shape (..., {self.hash_numbers}), but got {packed_codes.shape}"
            )
        if packed_codes.device != self.device:
            raise ValueError(
                f"packed_codes device {packed_codes.device} does not match required device {self.device}"
            )
        if packed_codes.dtype != torch.uint8:
            raise ValueError(
                f"packed_codes dtype {packed_codes.dtype} is not torch.uint8"
            )

        # e.g., packed_codes.shape=[s1, s2, s3, hash_numbers]
        # orig_shape = [s1, s2, s3]
        orig_shape = packed_codes.shape[:-1]

        # [N * hash_numbers], e.g., N = s1*s2*s3
        packed_codes_flat = packed_codes.view(-1)

        if self.device.type == "npu":
            # [N * hash_bits]
            unpacked_bits_flat = torch_npu.npu_sign_bits_unpack(
                packed_codes_flat, size=1, dtype=torch.float16
            )
        elif self.device.type == "cuda" or self.device.type == "cpu":
            # (TODO) improve performance later on CUDA ops and CPU SIMD instructions
            # [N, hash_numbers]
            packed_codes_2d = packed_codes_flat.view(-1, self.hash_numbers)

            # [N, hash_numbers, 8]
            expanded = packed_codes_2d.unsqueeze(-1).expand(
                -1, -1, 8
            )  # expand last dim to 8

            # (expanded & self.bit_masks) > 0 -> [N, hash_numbers, 8]
            unpacked_bits = (expanded & self.bit_masks) > 0

            # 0 -> -1, 1 -> 1
            unpacked_bits = unpacked_bits * 2 - 1

            unpacked_bits = unpacked_bits.to(torch.float16)

            # [N, hash_bits]
            unpacked_bits_flat = unpacked_bits.view(-1, self.hash_bits)
        else:
            raise ValueError(f"Unsupported device type: {self.device.type}")

        out_shape = orig_shape + (self.hash_bits,)
        unpacked_bits = unpacked_bits_flat.view(out_shape)

        return unpacked_bits


if __name__ == "__main__":
    dtype = torch.float16
    if hasattr(torch, "npu") and torch.npu.is_available():
        device = torch.device("npu:0")
    elif hasattr(torch, "cuda") and torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype=torch.float32
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    torch.manual_seed(42)

    encoder = HashEncoder(input_dim=8, hash_bits=8, dtype=dtype, device=device)

    x = torch.randn(2, 8, device=device, dtype=dtype)
    print("x:", x)

    hash_codes = encoder.compute_hash(x)
    print("hash_codes:", hash_codes)
    print("hash_codes shape:", hash_codes.shape)

    unpacked_bits = encoder._unpack_hash(hash_codes)
    print("unpacked_bits:", unpacked_bits)
    print("unpacked_bits shape:", unpacked_bits.shape)

    print(
        f"hash_codes[0].item()={hash_codes[0].item()}, 8-bit binary form:{hash_codes[0].item():08b}"
    )
    print(
        f"hash_codes[1].item()={hash_codes[1].item()}, 8-bit binary form:{hash_codes[1].item():08b}"
    )
