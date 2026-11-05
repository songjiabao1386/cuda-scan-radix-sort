#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include "radix_sort.cuh"

int main() {
    // 测试大规模乱序数据：100,000 个无符号整数
    int n = 100000;
    size_t bytes = n * sizeof(unsigned int);

    std::vector<unsigned int> h_in(n);
    std::vector<unsigned int> h_cpu_out(n);
    std::vector<unsigned int> h_gpu_out(n);

    // 随机数生成器
    std::mt19937 rng(42);
    std::uniform_int_distribution<unsigned int> dist(0, 1000000);

    for (int i = 0; i < n; ++i) {
        h_in[i] = dist(rng);
    }

    // 1. CPU 标准排序（黄金参考标准）
    h_cpu_out = h_in;
    std::sort(h_cpu_out.begin(), h_cpu_out.end());

    // 2. 分配 GPU 显存
    unsigned int *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    std::cout << "Running GPU Parallel Radix Sort for " << n << " elements..." << std::endl;
    
    // 3. 运行我们自己实现的 GPU 基数排序！
    radix_sort(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_gpu_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // 4. 对拍验证
    bool success = true;
    for (int i = 0; i < n; ++i) {
        if (h_gpu_out[i] != h_cpu_out[i]) {
            std::cerr << "Mismatch at index " << i << ": GPU=" << h_gpu_out[i] 
                      << ", CPU=" << h_cpu_out[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "SUCCESS: GPU Parallel Radix Sort perfectly matched with CPU std::sort!" << std::endl;
        std::cout << "First 10 sorted elements: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << h_gpu_out[i] << " ";
        }
        std::cout << "..." << std::endl;
    } else {
        std::cout << "FAILURE: Results do not match." << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
