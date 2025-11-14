#include <pybind11/pybind11.h>
#include <unordered_map>
#include <unordered_set>
#include <tuple>

namespace py = pybind11;

std::pair<std::unordered_map<int, int>, std::unordered_map<int, int>> diff_two_map(
    const std::unordered_map<int, int>& map1, const std::unordered_map<int, int>& map2) {
    
    std::unordered_set<int> keys2_set;
    std::unordered_set<int> values2_set;
    std::unordered_map<int, int> diff_map;
    std::unordered_map<int, int> updated_map;

    // 将 map2 的键和值存入集合
    for (const auto& entry : map2) {
        keys2_set.insert(entry.first);
        values2_set.insert(entry.second);
    }

    // 遍历 map1，查找存在于 map2 中的键值对
    for (const auto& entry1 : map1) {
        int k1 = entry1.first;
        int v1 = entry1.second;

        // 如果 map1 中的键值对在 map2 中也存在，则标记为更新
        if (keys2_set.find(k1) != keys2_set.end() && values2_set.find(v1) != values2_set.end()) {
            updated_map[k1] = v1;
            keys2_set.erase(k1);
            values2_set.erase(v1);
        }
    }

    // 将剩余的 map2 中的键值对添加到 diff_map 和 updated_map
    auto key_it = keys2_set.begin();
    auto value_it = values2_set.begin();
    while (key_it != keys2_set.end() && value_it != values2_set.end()) {
        int k2 = *key_it;
        int v2 = *value_it;

        diff_map[k2] = v2;
        updated_map[k2] = v2;

        ++key_it;
        ++value_it;
    }

    return {updated_map, diff_map};
}

int get_offset(const std::tuple<int, int, int>& block_shape, int rank, int tp_size, int precision, int layer_id, bool is_v, bool is_mla) {
    int block_size, num_key_heads_per_tp, head_size;
    std::tie(block_size, num_key_heads_per_tp, head_size) = block_shape;
    
    int k_min_data_block_size = block_size * num_key_heads_per_tp * head_size * precision;
    int v_min_data_block_size = is_mla ? 0 : k_min_data_block_size;
    int layer_size = k_min_data_block_size + v_min_data_block_size;
    
    int k_offset;
    if (is_mla) {
        k_offset = layer_size * layer_id;
    } else {
        layer_size *= tp_size;
        k_offset = layer_size * layer_id + layer_size / tp_size * rank;
    }
    
    int v_offset = k_offset + k_min_data_block_size;
    
    return is_v ? v_offset : k_offset;
}

PYBIND11_MODULE(kvcache_pre, m) {
    m.def("diff_two_map", &diff_two_map, "Compute the difference between two maps");
    m.def("get_offset", &get_offset, 
          "Calculate offset for KV cache retrieval", 
          py::arg("block_shape"), py::arg("rank"), py::arg("tp_size"), 
          py::arg("precision"), py::arg("layer_id"), py::arg("is_v"), py::arg("is_mla"));
}
