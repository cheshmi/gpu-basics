// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#ifndef VECTORIZATION_VEC_MUL_H
#define VECTORIZATION_VEC_MUL_H
#include <vector>

namespace swiftware::hpp {

  /// \brief Adds two vectors
  /// \param a The first vector
  /// \param b The second vector
  /// \param c The result vector
  void vec_mul(std::vector<float> a, std::vector<float> b, std::vector<float>& c);

  void vec_mul_unrolled8_scalarized(std::vector<float> a, std::vector<float> b, std::vector<float>& c);

  void vec_mul_unrolled_avx(std::vector<float> a, std::vector<float> b, std::vector<float>& c);

  void vec_mul_parallel(std::vector<float> a, std::vector<float> b, std::vector<float>& c, int num_threads);

  void vec_mul_unrolled8_scalarized_parallel(std::vector<float> a, std::vector<float> b, std::vector<float>& c, int num_threads);

  void vec_mul_unrolled_avx_parallel(std::vector<float> a, std::vector<float> b, std::vector<float>& c, int num_threads);
}

#endif //VECTORIZATION_VEC_MUL_H
