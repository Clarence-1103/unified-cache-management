# Clean up previous build if exists
rm -rf gpu_retrieval_backend.so

# Compile with explicit linking and C++ flags
nvcc -O3 -shared -std=c++17 \
    $(python3 -m pybind11 --includes) \
    gpu_retrieval_backend.cu \
    -Xcompiler -fPIC \
    -o gpu_retrieval_backend.so
