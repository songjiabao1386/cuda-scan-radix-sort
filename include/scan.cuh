#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// 1. 基础块级 Exclusive Scan 核函数（支持输出 Block Sums）
template <typename T>
__global__ void block_exclusive_scan_kernel(const T* d_in, T* d_out, T* d_block_sums, int n) {
    extern __shared__ char shared_mem[];
    T* temp = reinterpret_cast<T*>(shared_mem);

    int thid = threadIdx.x;
    int ai = thid;
    int bi = thid + blockDim.x;

    int global_ai = blockIdx.x * (blockDim.x * 2) + ai;
    int global_bi = blockIdx.x * (blockDim.x * 2) + bi;

    temp[ai] = (global_ai < n) ? d_in[global_ai] : 0;
    temp[bi] = (global_bi < n) ? d_in[global_bi] : 0;

    int offset = 1;
    // Up-sweep 阶段
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai_idx = offset * (2 * thid + 1) - 1;
            int bi_idx = offset * (2 * thid + 2) - 1;
            temp[bi_idx] += temp[ai_idx];
        }
        offset *= 2;
    }

    // 保存最后一个元素作为该 Block 的总和，并清零根节点准备 Down-sweep
    if (thid == 0) {
        if (d_block_sums != nullptr) {
            d_block_sums[blockIdx.x] = temp[2 * blockDim.x - 1];
        }
        temp[2 * blockDim.x - 1] = 0;
    }

    // Down-sweep 阶段
    for (int d = 1; d < 2 * blockDim.x; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai_idx = offset * (2 * thid + 1) - 1;
            int bi_idx = offset * (2 * thid + 2) - 1;
            T t = temp[ai_idx];
            temp[ai_idx] = temp[bi_idx];
            temp[bi_idx] += t;
        }
    }
    __syncthreads();

    // 写回全局内存
    if (global_ai < n) d_out[global_ai] = temp[ai];
    if (global_bi < n) d_out[global_bi] = temp[bi];
}

// 2. 累加核函数：将各 Block 的基准偏移量加回到对应区块的每个元素上
template <typename T>
__global__ void add_block_sums_kernel(T* d_out, const T* d_block_sums, int n) {
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int global_ai = idx;
    int global_bi = idx + blockDim.x;

    T block_sum = d_block_sums[blockIdx.x];

    if (global_ai < n) d_out[global_ai] += block_sum;
    if (global_bi < n) d_out[global_bi] += block_sum;
}

// 3. 主机端包装函数：支持任意长度的大规模全局 Exclusive Scan
template <typename T>
void global_exclusive_scan(const T* d_in, T* d_out, int n) {
    if (n <= 0) return;

    int threads = 512; // 每个 Block 512 个线程，处理 1024 个元素
    int elements_per_block = threads * 2;
    int blocks = (n + elements_per_block - 1) / elements_per_block;

    // 如果只有一个 Block，直接调用基础 Kernel 即可（显式转换为 T* 防止模板推导失败）
    if (blocks == 1) {
        size_t shared_mem_bytes = 2 * threads * sizeof(T);
        block_exclusive_scan_kernel<<<1, threads, shared_mem_bytes>>>(d_in, d_out, (T*)nullptr, n);
        return;
    }

    // 分配存储各个 Block 总和的显存
    T* d_block_sums = nullptr;
    T* d_scanned_block_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums, blocks * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_scanned_block_sums, blocks * sizeof(T)));

    // 阶段 1：计算每个 Block 的局部 Scan，并保存各自的总和到 d_block_sums
    size_t shared_mem_bytes = 2 * threads * sizeof(T);
    block_exclusive_scan_kernel<<<blocks, threads, shared_mem_bytes>>>(d_in, d_out, d_block_sums, n);

    // 阶段 2：对 Block 总和数组递归进行 Scan
    global_exclusive_scan(d_block_sums, d_scanned_block_sums, blocks);

    // 阶段 3：将扫描后的 Block 总和加回到对应 Block 的所有元素中
    add_block_sums_kernel<<<blocks, threads>>>(d_out, d_scanned_block_sums, n);

    // 释放中间显存
    cudaFree(d_block_sums);
    cudaFree(d_scanned_block_sums);
}
