# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/albakrih/Desktop/TA/2024/4sp4/gpu-basics/cmake-build-debug/_deps/googlebenchmark-src"
  "/home/albakrih/Desktop/TA/2024/4sp4/gpu-basics/cmake-build-debug/_deps/googlebenchmark-build"
  "/home/albakrih/Desktop/TA/2024/4sp4/gpu-basics/cmake-build-debug/_deps/googlebenchmark-subbuild/googlebenchmark-populate-prefix"
  "/home/albakrih/Desktop/TA/2024/4sp4/gpu-basics/cmake-build-debug/_deps/googlebenchmark-subbuild/googlebenchmark-populate-prefix/tmp"
  "/home/albakrih/Desktop/TA/2024/4sp4/gpu-basics/cmake-build-debug/_deps/googlebenchmark-subbuild/googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp"
  "/home/albakrih/Desktop/TA/2024/4sp4/gpu-basics/cmake-build-debug/_deps/googlebenchmark-subbuild/googlebenchmark-populate-prefix/src"
  "/home/albakrih/Desktop/TA/2024/4sp4/gpu-basics/cmake-build-debug/_deps/googlebenchmark-subbuild/googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/albakrih/Desktop/TA/2024/4sp4/gpu-basics/cmake-build-debug/_deps/googlebenchmark-subbuild/googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/albakrih/Desktop/TA/2024/4sp4/gpu-basics/cmake-build-debug/_deps/googlebenchmark-subbuild/googlebenchmark-populate-prefix/src/googlebenchmark-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
