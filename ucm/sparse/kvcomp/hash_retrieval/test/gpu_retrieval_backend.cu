// retrieval_backend.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <random>
#ifdef NUMA_ENABLED
#include <numaif.h>
#endif
#include <iostream>

namespace py = pybind11;

#include <cuda_runtime.h>
#include <cub/cub.cuh>

// 定义一些 CUDA 错误检查宏
#define cuda_check(call){ \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        fprintf(stderr, "cuda error %s %d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
}

__global__ void compute_scores(
    const float* __restrict__ queries,       // [batch, dim_]
    const float* __restrict__ data,          // [num_data, dim_]
    const int* __restrict__ allowed,         // [batch, allowed_size] - flattened
    float* __restrict__ scores,              // [batch * topk]
    int batch,                               // 批次大小
    int dim_,                                // 查询维度
    int allowed_size                         // 允许查询数据集大小
) {
    // 使用二维网格来处理查询和 allowed 索引
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;  // 当前线程计算哪个查询
    int allowed_idx = blockIdx.y * blockDim.y + threadIdx.y; // 当前线程计算哪个 allowed 索引

    if (query_idx < batch && allowed_idx < allowed_size) {
        const float* q_ptr = queries + query_idx * dim_;   // 当前查询的指针
        int idx = allowed[query_idx * allowed_size + allowed_idx];  // 对应 allowed 中的索引
        
        float score = 0.0f;
        // 计算查询与数据点的内积
        #pragma unroll 8
        for (int d = 0; d < dim_; ++d) {
            score += q_ptr[d] * data[idx * dim_ + d];
        }
        
        // 将结果存储到 scores 数组
        scores[query_idx * allowed_size + allowed_idx] = score;
    }
}

class RetrievalWorkerBackend {
public:
    RetrievalWorkerBackend(py::array_t<float> data,
                           py::dict cpu_idx_tbl) 
        : data_array_(data), stop_workers_(false), next_req_id_(0)
    {
        py::buffer_info info = data_array_.request();
        n_items_ = info.shape[0];
        dim_ = info.shape[1];
        data_ = static_cast<const float*>(info.ptr);

        // Start worker threads
        for (auto cpu_idx : cpu_idx_tbl) {
            py::list core_ids = cpu_idx.second.cast<py::list>();

            for (size_t i = 0; i < core_ids.size(); ++i) {
                int core_id = core_ids[i].cast<int>();
                worker_threads_.emplace_back(&RetrievalWorkerBackend::worker_loop, this);

                // 核心绑定代码
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(core_id, &cpuset);  // 绑定每个线程到指定的核心

                pthread_t thread = worker_threads_.back().native_handle();
                
                // 设置 CPU 亲和性
                int rc = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
                if (rc != 0) {
                    std::cerr << "Error binding thread " << i << " to CPU core " << core_id << std::endl;
                }

            #ifdef NUMA_ENABLED
                int numaId = cpu_idx.first.cast<int>();
                // 设置内存亲和性
                unsigned long nodeMask = 1UL << numaId;
                rc = set_mempolicy(MPOL_BIND, &nodeMask, sizeof(nodeMask) * 8);
                if (rc != 0) {
                    std::cerr << "Error binding memory to NUMA node " << numaId << std::endl;
                }
            #endif
            }

        }
    }

    ~RetrievalWorkerBackend() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_workers_ = true;
            cond_.notify_all();
        }
        for (auto& t: worker_threads_) t.join();
    }

    int submit(py::array_t<float> query, int topk, py::array_t<int> indexes) {
        py::buffer_info qinfo = query.request();
        py::buffer_info iinfo = indexes.request();
        if (qinfo.shape[1] != dim_)
            throw std::runtime_error("Query dim mismatch");
        if ((size_t)iinfo.shape[0] != (size_t)qinfo.shape[0])
            throw std::runtime_error("Query and indexes batch mismatch");

        int req_id = next_req_id_.fetch_add(1);

        auto q = std::vector<float>((float*)qinfo.ptr, (float*)qinfo.ptr + qinfo.shape[0] * dim_);

        // Parse indexes to vector<vector<int>>
        size_t n_requests = iinfo.shape[0], max_index_number = iinfo.shape[1];
        const int* idx_ptr = static_cast<const int*>(iinfo.ptr);
        std::vector<std::vector<int>> idxvec(n_requests);
        for (size_t i = 0; i < n_requests; ++i) {
            for (size_t j = 0; j < max_index_number; ++j) {
                int index = idx_ptr[i * max_index_number + j];
                if (index != -1) idxvec[i].push_back(index);
            }
        }

        auto status = std::make_shared<RequestStatus>();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            requests_.emplace(Request{req_id, std::move(q), n_requests, topk, std::move(idxvec)});
            request_status_[req_id] = status;
        }
        cond_.notify_one();
        return req_id;
    }

    bool poll(int req_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        return results_.find(req_id) != results_.end();
    }

    void wait(int req_id) {
        std::shared_ptr<RequestStatus> s;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = request_status_.find(req_id);
            if (it == request_status_.end()) throw std::runtime_error("Bad req_id");
            s = it->second;
        }
        std::unique_lock<std::mutex> lk2(s->m);
        s->cv.wait(lk2, [&] { return s->done; });
    }

    py::dict get_result(int req_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = results_.find(req_id);
        if (it == results_.end()) throw std::runtime_error("Result not ready");

        size_t batch_size = it->second.indices.size();
        size_t topk = batch_size > 0 ? it->second.indices[0].size() : 0;
        py::array_t<int> indices({batch_size, topk});
        py::array_t<float> scores({batch_size, topk});

        auto indices_ptr = static_cast<int*>(indices.request().ptr);
        auto scores_ptr = static_cast<float*>(scores.request().ptr);

        for (size_t i = 0; i < batch_size; ++i) {
            memcpy(indices_ptr + i * topk, it->second.indices[i].data(), topk * sizeof(int));
            memcpy(scores_ptr + i * topk, it->second.scores[i].data(), topk * sizeof(float));
        }
        py::dict result;
        result["indices"] = indices;
        result["scores"] = scores;
        results_.erase(it);
        return result;
    }

private:
    struct Request {
        int req_id;
        std::vector<float> query; // Flattened [batch, dim]
        size_t batch;
        int topk;
        std::vector<std::vector<int>> indexes; // Per-request index subset
    };
    struct Result {
        std::vector<std::vector<int>> indices;
        std::vector<std::vector<float>> scores;
    };
    struct RequestStatus {
        std::mutex m;
        std::condition_variable cv;
        bool done = false;
    };

    void worker_loop() {
        while (true) {
            Request req;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cond_.wait(lock, [&]{ return stop_workers_ || !requests_.empty(); });
                if (stop_workers_ && requests_.empty()) return;
                req = std::move(requests_.front());
                requests_.pop();
            }

            Result res;
            res.indices.resize(req.batch);
            res.scores.resize(req.batch);

            int batch = req.batch;  // 批次大小
            int allowed_size = req.indexes[0].size();  // 每个查询的允许数据大小
            int topk = req.topk;  // Top-K 数量

            // 1. 分配内存
            float* d_queries;
            float* d_data;
            float* d_scores;
            int*   d_allowed;
            int*   d_sorted_indices;
            float* d_sorted_scores;

            cuda_check(cudaMalloc(&d_queries, batch * dim_ * sizeof(float)));
            cuda_check(cudaMalloc(&d_data, n_items_ * dim_ * sizeof(float)));
            cuda_check(cudaMalloc(&d_scores, batch * allowed_size * sizeof(float)));
            cuda_check(cudaMalloc(&d_allowed, batch * allowed_size * sizeof(int)));
            cuda_check(cudaMalloc(&d_sorted_indices, batch * allowed_size * sizeof(int)));
            cuda_check(cudaMalloc(&d_sorted_scores, batch * allowed_size * sizeof(float)));

            // 2. 将数据从主机复制到设备
            cuda_check(cudaMemcpy(d_queries, req.query.data(), batch * dim_ * sizeof(float), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(d_data, data_, n_items_ * dim_ * sizeof(float), cudaMemcpyHostToDevice));
            cuda_check(cudaMemcpy(d_allowed, req.indexes.data(), batch * allowed_size * sizeof(int), cudaMemcpyHostToDevice));

            // 3. 计算相似度
            int threads_per_block = 256;
            int blocks = (batch + threads_per_block - 1) / threads_per_block;
            compute_scores<<<blocks, threads_per_block>>>(d_queries, d_data, d_allowed, d_scores, batch, dim_, allowed_size);
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("CUDA compute scores error: %s\n", cudaGetErrorString(err));
            }

            // 4.1 run CUB segmented radix sort (one segment)
            // 4.2 run CUB segmented radix sort over B batches
            std::vector<int> h_offsets(batch + 1);
            for (int i = 0; i <= B; ++i) h_offsets[i] = i * allowed_size;
            int* d_offsets;
            cuda_check(cudaMalloc(&d_offsets, (batch + 1) * sizeof(int)));
            cuda_check(cudaMemcpy(d_offsets, h_offsets.data(), (batch + 1) * sizeof(int), cudaMemcpyHostToDevice));
            void*  d_temp = nullptr;
            size_t temp_bytes = 0;
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                d_temp, temp_bytes,
                d_scores,  d_sorted_scores,
                d_allowed, d_sorted_indices,
                batch * allowed_size, batch, d_offsets, d_offsets + 1);
            cuda_check(cudaMalloc(&d_temp, temp_bytes));
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                d_temp, temp_bytes,
                d_scores,  d_sorted_scores,
                d_allowed, d_sorted_indices,
                batch * allowed_size, batch, d_offsets, d_offsets + 1);

            // 5) copy top-K indices back
            for (int k = 0; k < curr_topk; ++k) {
                    res.scores[b].push_back(heap[k].first);
                    res.indices[b].push_back(heap[k].second);
                }
            
            Result res;
            res.indices.resize(req.batch);
            res.scores.resize(req.batch);
            
            cuda_check(cudaMemcpy(h_topk.data(), d_sorted_indices, req.batch * allowed_size * sizeof(int), cudaMemcpyDeviceToHost));
            // cuda_check(cudaMemcpy(h_topk.data(), d_sorted_idx, B * K * sizeof(int), cudaMemcpyDeviceToHost));


            // 5. 复制结果回主机
            cuda_check(cudaMemcpy(res.indices.data(), d_topk_indices, batch * topk * sizeof(int), cudaMemcpyDeviceToHost));
            cuda_check(cudaMemcpy(res.scores.data(), d_topk_scores, batch * topk * sizeof(float), cudaMemcpyDeviceToHost));


            // 清理设备内存
            cuda_check(cudaFree(d_queries));
            cuda_check(cudaFree(d_data));
            cuda_check(cudaFree(d_scores));
            cuda_check(cudaFree(d_allowed));
            cuda_check(cudaFree(d_topk_indices));
            cuda_check(cudaFree(d_topk_scores));

            // 将计算结果存储并通知状态
            {
                std::lock_guard<std::mutex> lock(mutex_);
                results_[req.req_id] = std::move(res);
                auto s = request_status_[req.req_id];
                {
                    std::lock_guard<std::mutex> lk2(s->m);
                    s->done = true;
                }
                s->cv.notify_all();
            }
        }
    }


    py::array_t<float> data_array_;
    const float* data_ = nullptr;
    ssize_t n_items_, dim_;
    std::queue<Request> requests_;
    std::unordered_map<int, Result> results_;
    std::vector<std::thread> worker_threads_;
    std::mutex mutex_;
    std::condition_variable cond_;
    std::unordered_map<int, std::shared_ptr<RequestStatus>> request_status_;
    bool stop_workers_;
    std::atomic<int> next_req_id_;
};

PYBIND11_MODULE(gpu_retrieval_backend, m) {
    py::class_<RetrievalWorkerBackend>(m, "RetrievalWorkerBackend")
        .def(py::init<py::array_t<float>, py::dict>())
        .def("submit", &RetrievalWorkerBackend::submit)
        .def("poll", &RetrievalWorkerBackend::poll)
        .def("get_result", &RetrievalWorkerBackend::get_result)
        .def("wait", &RetrievalWorkerBackend::wait);
}
