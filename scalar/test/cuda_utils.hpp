#ifndef CUDA_TRIDSOLV_TEST_UTILS_HPP
#define CUDA_TRIDSOLV_TEST_UTILS_HPP

#include "utils.hpp"


template <typename Float> class GPUMesh {
  std::vector<int> _dims;
  Float *_a_d, *_b_d, *_c_d, *_d_d;

public:
  GPUMesh(const MeshLoader<Float> &mesh);
  ~GPUMesh();

  const std::vector<int> &dims() const { return _dims; }
  const Float *a() const { return _a_d; }
  const Float *b() const { return _b_d; }
  const Float *c() const { return _c_d; }
  Float *d() const { return _d_d; }
};

/**********************************************************************
 *                          Implementations                           *
 **********************************************************************/

template <typename Float>
GPUMesh<Float>::GPUMesh(const MeshLoader<Float> &mesh)
    : _dims(mesh.dims()) {
  size_t capacity = 1;
  for (size_t d : _dims) {
    capacity *= d;
  }
  cudaMalloc((void**)&_a_d, sizeof(Float) * capacity);
  cudaMalloc((void**)&_b_d, sizeof(Float) * capacity);
  cudaMalloc((void**)&_c_d, sizeof(Float) * capacity);
  cudaMalloc((void**)&_d_d, sizeof(Float) * capacity);
  cudaMemcpy(_a_d, mesh.a().data(), capacity * sizeof(Float), cudaMemcpyHostToDevice);
  cudaMemcpy(_b_d, mesh.b().data(), capacity * sizeof(Float), cudaMemcpyHostToDevice);
  cudaMemcpy(_c_d, mesh.c().data(), capacity * sizeof(Float), cudaMemcpyHostToDevice);
  cudaMemcpy(_d_d, mesh.d().data(), capacity * sizeof(Float), cudaMemcpyHostToDevice);
}

template <typename Float> GPUMesh<Float>::~GPUMesh() {
  cudaFree(_a_d);
  cudaFree(_b_d);
  cudaFree(_c_d);
  cudaFree(_d_d);
}

#endif /* ifndef CUDA_TRIDSOLV_TEST_UTILS_HPP */
