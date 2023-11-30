#include <iostream>
#include <chrono>
#include <stdexcept>
#include <sstream>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>

#define CUDA_CHECK_ERROR(status) cuda_error_check(status, __FILE__, __LINE__, __func__)

namespace {
inline void cuda_error_check(cudaError_t error, const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
  if(error != cudaSuccess){
    std::stringstream ss;
    ss << cudaGetErrorString( error );
    ss <<" [" << filename << ":" << line << " in " << funcname << "]";
    throw std::runtime_error(ss.str());
  }
}

template <class T>
void gemv(
    cublasHandle_t const handle,
    const std::size_t m,
    const std::size_t n,
    const T* const A_ptr,
    const T* const x_ptr,
    T* const y_ptr
    );
template <>
void gemv<double>(
    cublasHandle_t const handle,
    const std::size_t m,
    const std::size_t n,
    const double* const A_ptr,
    const double* const x_ptr,
    double* const y_ptr
    ) {
  double one = 1, zero = 0;
  cublasDgemv(
      handle,
      CUBLAS_OP_N,
      m, n,
      &one,
      A_ptr, m,
      x_ptr, 1,
      &zero,
      y_ptr, 1
      );
}
template <class T>
void gemv_batched(
    cublasHandle_t const handle,
    const std::size_t m,
    const std::size_t n,
    const T* const A_ptr,
    const T* const x_ptr,
    T* const y_ptr,
    const std::size_t batch_size
    );
template <>
void gemv_batched<double>(
    cublasHandle_t const handle,
    const std::size_t m,
    const std::size_t n,
    const double* const A_ptr,
    const double* const x_ptr,
    double* const y_ptr,
    const std::size_t batch_size
    ) {
  double one = 1, zero = 0;
  cublasDgemvStridedBatched(
      handle,
      CUBLAS_OP_N,
      m, n,
      &one,
      A_ptr, m, m * n,
      x_ptr, 1, n,
      &zero,
      y_ptr, 1, m,
      batch_size
      );
}

template <class T>
struct GemvBase {
  using dtype = T;
  const std::string name;

  GemvBase(const std::string name) : name(name) {}

  virtual void operator() (
    const std::size_t m,
    const std::size_t n,
    const double* const A_ptr,
    const double* const x_ptr,
    double* const y_ptr,
    const std::size_t batch_size
      ) = 0;
};

template <class T>
struct BatchedGemv : public GemvBase<T> {
  cublasHandle_t cublas_handle;
  BatchedGemv() : GemvBase<T>("BatchedGemv") {
    cublasCreate(&cublas_handle);
  }

  void operator() (
    const std::size_t m,
    const std::size_t n,
    const T* const A_ptr,
    const T* const x_ptr,
    T* const y_ptr,
    const std::size_t batch_size
      ) {
    gemv_batched(
        cublas_handle,
        m, n,
        A_ptr,
        x_ptr,
        y_ptr,
        batch_size
        );
  }
};

template <unsigned num_streams, class T>
struct StreamedGemv : public GemvBase<T> {
  cudaStream_t stream_list[num_streams];
  cublasHandle_t cublas_handle_list[num_streams];
  StreamedGemv() : GemvBase<T>("StreamedGemv") {
    for (std::size_t i = 0; i < num_streams; i++) {
      cublasCreate(&cublas_handle_list[i]);
      cudaStreamCreate(&stream_list[i]);
      cublasSetStream(cublas_handle_list[i], stream_list[i]);
    }
  }
  void operator() (
    const std::size_t m,
    const std::size_t n,
    const double* const A_ptr,
    const double* const x_ptr,
    double* const y_ptr,
    const std::size_t batch_size
      ) {
    for (std::size_t i = 0; i < batch_size; i++) {
      gemv(
          cublas_handle_list[i % num_streams],
          m, n,
          A_ptr + i * m * n,
          x_ptr + i * n,
          y_ptr + i * m
          );
    }
  }
};

template <class MultiGemvT>
void eval(
    const std::size_t m,
    const std::size_t n,
    const std::size_t batch_size
    ) {
  using T = typename MultiGemvT::dtype;

  // Alloc
  T *d_A_ptr, *d_x_ptr, *d_y_ptr;
  CUDA_CHECK_ERROR(cudaMalloc(&d_A_ptr, m * n * batch_size * sizeof(T)));
  CUDA_CHECK_ERROR(cudaMalloc(&d_x_ptr, n * batch_size * sizeof(T)));
  CUDA_CHECK_ERROR(cudaMalloc(&d_y_ptr, m * batch_size * sizeof(T)));

  // Init
  CUDA_CHECK_ERROR(cudaMemset(d_A_ptr, 0, m * n * batch_size * sizeof(T)));
  CUDA_CHECK_ERROR(cudaMemset(d_x_ptr, 0, n * batch_size * sizeof(T)));
  CUDA_CHECK_ERROR(cudaMemset(d_y_ptr, 0, m * batch_size * sizeof(T)));

  MultiGemvT multi_gemv;
  multi_gemv(
      m, n,
      d_A_ptr,
      d_x_ptr,
      d_y_ptr,
      batch_size
      );

  const auto test_count = 10;
  cudaDeviceSynchronize();
  const auto start_clock = std::chrono::system_clock::now();
  for (std::size_t i = 0; i < test_count; i++) {
    multi_gemv(
        m, n,
        d_A_ptr,
        d_x_ptr,
        d_y_ptr,
        batch_size
        );
  }
  cudaDeviceSynchronize();
  const auto end_clock = std::chrono::system_clock::now();
  const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock).count() * 1e-9 / test_count;
  const auto ldst_data_size = (m * n + m + n) * sizeof(T) * batch_size;
  const auto flops = 2 * m * n * batch_size;
  std::printf(
      "imp=%12s, m=%4lu, n=%4lu, batch_size=%4lu, time=%.4e, achieved_bw=%.4e, achieved_gflops=%.4e\n",
      multi_gemv.name.c_str(),
      m, n,
      batch_size,
      time,
      ldst_data_size / time * 1e-9,
      flops / time * 1e-9
      );
  std::fflush(stdout);

  CUDA_CHECK_ERROR(cudaFree(d_A_ptr));
  CUDA_CHECK_ERROR(cudaFree(d_x_ptr));
  CUDA_CHECK_ERROR(cudaFree(d_y_ptr));
}

} // namespace

int main() {
  for (std::size_t i = 1; i < 20; i++) {
    const auto n = i * 32;
    const auto batch_size = 500;
    eval<BatchedGemv<double>>(n, n, batch_size);
    eval<StreamedGemv<16, double>>(n, n, batch_size);
  }
}
