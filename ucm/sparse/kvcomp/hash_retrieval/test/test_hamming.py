import numpy as np
import torch
import time
import hash_retrieval_backend

class HashRetrievalWorker:
    # handle torch -> numpy && float16/bfloat16 -> float32.
    def __init__(self, cpp_worker):
        self.cpp_worker = cpp_worker

    def submit(self, query, topk, indexes):
        req_id = self.cpp_worker.submit(query, topk, indexes)
        return req_id

    def poll(self, req_id):
        return self.cpp_worker.poll(req_id)  # Returns True if ready

    def get_result(self, req_id):
        return self.cpp_worker.get_result(req_id)

    def wait(self, req_id):
        return self.cpp_worker.wait(req_id)
    
if __name__ == "__main__":
    ################# data
    np.random.seed(42)

    req_batch = 100

    sum_time = 0.0
    for req in range(req_batch):
        batch_size = 1
        block_size = 128
        dim = 128
        kv_cache_blocks = 8000
        data = np.random.uniform(0, 255, (kv_cache_blocks, block_size, dim)).astype(np.uint8)

        backend = hash_retrieval_backend.HashRetrievalWorkerBackend(data)
        worker = HashRetrievalWorker(backend)
        topk = 15
        search_blocks_range = 1000
        tpot = 15 / 1000
        
        indexes = np.random.randint(0, kv_cache_blocks, (batch_size, search_blocks_range))
        query = np.random.uniform(0, 255, (batch_size, dim)).astype(np.uint8)

        start_time = time.time()
        req_id = worker.submit(query, topk=topk, indexes=indexes)
        worker.wait(req_id)
        result = worker.get_result(req_id)
        
        sum_time += (time.time() - start_time)
    
    print(f"old hash retrieval backend spent {sum_time} s")
