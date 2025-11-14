# 添加编译目标
pybind11_add_module(kvcache_pre cpy/kvcache_pre.cpp)

# 设置输出库的目录
set_target_properties(kvcache_pre PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
