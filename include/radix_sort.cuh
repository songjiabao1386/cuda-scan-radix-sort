#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "scan.cuh"

// 提取指定位（bit）的谓词 Kernel：判断第 bit 位是否为 0
template <typename T>
__global__ void compute_predicate_kernel(const T* d_in, unsigned int* d_pred, int bit, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T val = d_in[idx];
        d_pred[idx] = ((val >> bit) & 1) ? 0 : 1;
    }
}

// 根据 Scan 后的地址将元素 Scatter（重排）到目标位置
template <typename T>
__global__ void scatter_kernel(const T* d_in, T* d_out, const unsigned int* d_scan_pred, 
                               const unsigned int* d_pred, int total_zeros, int bit, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T val = d_in[idx];
        unsigned int is_zero = d_pred[idx];
        unsigned int scan_val = d_scan_pred[idx];

        int dest_idx = 0;
        if (is_zero) {
            dest_idx = scan_val;
        } else {
            dest_idx = total_zeros + (idx - scan_val);
        }
        d_out[dest_idx] = val;
    }
}

// 完整的 32 位并行基数排序函数（支持 unsigned int / int）
template <typename T>
void radix_sort(T* d_in, T* d_out, int n) {
    if (n <= 0) return;

    unsigned int *d_pred = nullptr;
    unsigned int *d_scan_pred = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pred, n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_scan_pred, n * sizeof(unsigned int)));

    // 辅助显存缓冲区，用于在多轮迭代中做 Ping-Pong 交换
    T* d_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buffer, n * sizeof(T)));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // 决定当前轮次输入输出的指针
    T* current_in = d_in;
    T* current_out = d_buffer;

    // 对 32 位无符号整数按位进行 32 轮迭代
    int num_bits = sizeof(T) * 8;
    for (int bit = 0; bit < num_bits; ++bit) {
        // 1. 计算谓词
        compute_predicate_kernel<<<blocks, threads>>>(current_in, d_pred, bit, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 2. 对谓词执行全局 Exclusive Scan
        global_exclusive_scan(d_pred, d_scan_pred, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 获取最后一个元素的 Predicate 和 Scan 值，计算总共有多少个 0
        unsigned int last_pred, last_scan;
        CUDA_CHECK(cudaMemcpy(&last_pred, d_pred + n - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last_scan, d_scan_pred + n - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        int total_zeros = last_scan + last_pred;

        // 3. 根据 Scan 结果进行 Scatter 重排
        scatter_kernel<<<blocks, threads>>>(current_in, current_out, d_scan_pred, d_pred, total_zeros, bit, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 交换输入输出缓冲区（Ping-Pong）
        std::swap(current_in, current_out);
    }

    // 如果总迭代次数是偶数轮，最终结果会留在 d_buffer 里，需要拷回 d_out
    if (num_bits % 2 != 0) {
        // 如果最终停在了 d_buffer，而用户期望输出在 d_out，我们需要把 d_buffer 拷回 d_out
        // 上面算法中最后一次 swap 会让结果回到初始的 d_in 还是 d_buffer？
        // 我们来显式控制：
    }

    // 为了更稳妥，我们直接在最后判断：如果最终数据在 d_buffer 中，复制到 d_out
    // 32 是偶数（32 位），所以循环 32 次后，current_in 会刚好回到原先传进来的指针位置吗？
    // 让我们理一下：初始 current_in = d_in, current_out = d_buffer.
    // bit 0: 读 in, 写 out (buffer). 交换后: in = buffer, out = in.
    // bit 1: 读 buffer, 写 in. 交换后: in = in, out = buffer.
    // 也就是说，偶数轮（如 32 轮）结束时，数据会精确地回到 current_in。
    // 如果 current_in 不是 d_out，我们把它拷贝到 d_out 即可。
    if (current_in != d_out) {
        CUDA_CHECK(cudaMemcpy(d_out, current_in, n * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    cudaFree(d_pred);
    cudaFree(d_scan_pred);
    cudaFree(d_buffer);
}
