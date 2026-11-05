# CUDA High-Performance Parallel Scan & Radix Sort

[![CUDA 11.0+](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight, high-performance GPU parallel computing library implemented from scratch in **CUDA C++**, mirroring key primitives of NVIDIA Thrust. 

This project demonstrates deep hardware-level GPU programming, utilizing **shared memory optimizations**, **block-level synchronization**, and **work-efficient tree-based reduction algorithms** to accelerate foundational HPC primitives.

---

## 🌟 Key Features

* **Multi-Block Global Exclusive Scan (Prefix Sum)**
  * Implements the work-efficient **Blelloch Scan algorithm** (Up-sweep & Down-sweep) leveraging high-speed **Shared Memory**.
  * Scales seamlessly beyond single-block limits via a **Two-Phase Recursive Global Scan** architecture, supporting arbitrary-sized datasets ($100,000+$ elements and beyond).
  * Includes dedicated block-sum reduction and parallel add-back kernels for thread-safe global offset distribution.
* **32-Bit Parallel Radix Sort**
  * Built directly on top of the high-performance Exclusive Scan primitive.
  * Features **Predicate Stream Compaction** and parallel **Scatter** routing.
  * Utilizes an efficient **Ping-Pong Buffer** swapping mechanism to iterate through all 32 bits with zero extra allocation overhead during execution.
* **Automated Verification & Golden Reference Testing**
  * Built-in CPU reference verification suite.
  * $100\%$ accuracy validation against CPU reference scan and `std::sort` across large-scale random integer datasets.

---

## 📐 Architecture & Algorithm Overview

### 1. Global Exclusive Scan (Blelloch Algorithm)
The scan algorithm runs in two primary phases per Block, followed by a hierarchical global sum aggregation:

1. **Up-sweep (Reduce)**: Recursively sums pair-wise elements up a binary tree in Shared Memory.
2. **Down-sweep**: Clears the root element and traverses back down the tree to build exclusive prefixes.
3. **Global Multi-Block Aggregation**:
   $$\text{Block Scan} \longrightarrow \text{Recursive Scan on Block Sums} \longrightarrow \text{Parallel Add-Back}$$

```
Input Array:     [ 1,  2,  3,  4 |  5,  6,  7,  8 ]  (Split into Blocks)
Phase 1 (Local): [ 0,  1,  3,  6 |  0,  5, 11, 18 ]  (Block Sums: [10, 26])
Phase 2 (Scan):  [ 0, 10 ]                           (Scanned Block Sums)
Phase 3 (Add):   [ 0,  1,  3,  6 | 10, 15, 21, 28 ]  (Final Exclusive Scan Result)
```

### 2. Radix Sort Execution Flow
For each bit ($0 \dots 31$), elements are partitioned into `0`-buckets and `1`-buckets:
1. **Predicate Compute**: Check if bit $k$ of each element is $0$.
2. **Scan Predicate**: Calculate destination offsets for `0` elements.
3. **Scatter**: Write `0` elements to `scan_val` index, and `1` elements to `total_zeros + (i - scan_val)`.
4. **Ping-Pong Swap**: Swap input/output device pointers for the next bit iteration.

---

## 📂 Project Structure

```text
gpu_scan_radix_sort/
├── CMakeLists.txt          # CMake build configuration
├── include/
│   ├── scan.cuh            # Multi-block Blelloch Exclusive Scan Implementation
│   └── radix_sort.cuh      # 32-bit Parallel Radix Sort & Predicate Kernels
└── src/
    └── main.cu             # Automated validation & CPU cross-check suite
```

---

## 🛠️ Build & Run

### Prerequisites
* **NVIDIA CUDA Toolkit** (NVCC 11.0 or higher)
* **CMake** (v3.18 or higher)
* C++17 compatible host compiler (GCC/Clang/MSVC)

### Compilation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cuda-scan-radix-sort.git
cd cuda-scan-radix-sort

# Create build directory and compile
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Running Test Suite

```bash
./gpu_test
```

**Expected Sample Output**:
```text
Running GPU Parallel Radix Sort for 100000 elements...
SUCCESS: GPU Parallel Radix Sort perfectly matched with CPU std::sort!
First 10 sorted elements: 5 9 11 16 20 30 48 52 67 93 ...
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.