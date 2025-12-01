rm -rf hash_retrieval_backend.so

c++ -O3 -march=native -fopenmp -g3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    hash_retrieval_backend.cpp \
    -o hash_retrieval_backend.so 

