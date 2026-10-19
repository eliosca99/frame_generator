#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) \
            + " (" + __FILE__ + ":" + std::to_string(__LINE__) + ")"); \
        } \
    } while (0)

// divisione per eccesso: dice quanti blocchi di dimensione b servono per contenere a elementi
inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

template <typename T>
class DeviceBuffer {
    T* ptr_ = nullptr;
    size_t n_ = 0;
    public:
        explicit DeviceBuffer(size_t n): n_(n) {
            CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
        }
        ~DeviceBuffer() {
            if(ptr_) cudaFree(ptr_);
        }
        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
        T* get() const { return ptr_; }
        void upload(const T* host) { CUDA_CHECK(cudaMemcpy(ptr_, host, n_ * sizeof(T), cudaMemcpyHostToDevice)); }
        void download(T* host) const { CUDA_CHECK(cudaMemcpy(host, ptr_, n_ * sizeof(T), cudaMemcpyDeviceToHost)); }
};

class GpuTimer {
    cudaEvent_t a_, b_;

    public:
        GpuTimer() { cudaEventCreate(&a_); cudaEventCreate(&b_); }
        ~GpuTimer() { cudaEventDestroy(a_); cudaEventDestroy(b_); }

        void start() { cudaEventRecord(a_); }

        float stop() {
            cudaEventRecord(b_);
            cudaEventSynchronize(b_);
            float ms = 0;
            cudaEventElapsedTime(&ms, a_, b_);
            return ms;
        }
};

void warmup() { CUDA_CHECK(cudaFree(nullptr)); }