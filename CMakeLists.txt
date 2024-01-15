cmake_minimum_required(VERSION 3.8)
project(cuda_basics)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc) # Location of the CUDA Toolkit
# Set CMP0104 policy to NEW to define CUDA architectures
# cmake_policy(SET CMP0104 NEW)

set(CUDA_ENABLE ON CACHE BOOL "enable NVCC compiler")
set(CUDA_VERBOSE_BUILD OFF CACHE BOOL "nvcc verbose" FORCE)
set(CUDA_NVCC_FLAGS -Wno-deprecated-gpu-targets CACHE STRING "suppress deprecated warnings" FORCE)
set(CMAKE_CUDA_ARCHITECTURES 70) # Define CUDA architectures

message(STATUS "CUDA_FOUND is ${CUDA_FOUND}")
message(STATUS "CUDA_ENABLE is ${CUDA_ENABLE}")
message(STATUS "CUDA_VERSION_STRING is ${CUDA_VERSION_STRING}")
message(STATUS "CUDA_VERBOSE_BUILD is ${CUDA_VERBOSE_BUILD}")
message(STATUS "CUDA_NVCC_HOST_COMPILER is ${CUDA_NVCC_HOST_COMPILER}")
message(STATUS "CUDA_PROPAGATE_HOST_FLAGS is ${CUDA_PROPAGATE_HOST_FLAGS}")
message(STATUS "CUDA_TOOLKIT_ROOT_DIR is ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDAToolkit_VERSION is ${CUDAToolkit_VERSION}")
message(STATUS "CUDA_INCLUDE_DIRS is ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES is ${CUDA_LIBRARIES}")
message(STATUS "CUDAToolkit_INCLUDE_DIRS is ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDAToolkit_LIBRARY_DIR is ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "CUDAToolkit_BIN_DIR is ${CUDAToolkit_BIN_DIR}")
message(STATUS "CUDA_CUBLAS_LIBRARIES is ${CUDA_CUBLAS_LIBRARIES}")
message(STATUS "CUDA_CUFFT_LIBRARIES is ${CUDA_CUFFT_LIBRARIES}")
message(STATUS "CUDA_CURAND_LIBRARIES is ${CUDA_CURAND_LIBRARIES}")
message(STATUS "CUDA_CUSOLVER_LIBRARIES is ${CUDA_CUSOLVER_LIBRARIES}")
message(STATUS "CUDA_CUSPARSE_LIBRARIES is ${CUDA_CUSPARSE_LIBRARIES}")
message(STATUS "CUDA_NPP_LIBRARIES is ${CUDA_NPP_LIBRARIES}")
message(STATUS "CUDA_OPENCL_LIBRARIES is ${CUDA_OPENCL_LIBRARIES}")

set(CUDA_CURAND_LIBRARIES ${CUDAToolkit_LIBRARY_DIR}/libcurand.so)

include_directories(
  include
  ${CUDA_INCLUDE_DIRS}
)

# Executables output directory
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

cuda_add_executable(vectoradd vectoradd.cu)
cuda_add_executable(matrixmult matrixmult.cu)
cuda_add_executable(vectorsum vectorsum.cu)
cuda_add_executable(convolution1d convolution1d.cu)
cuda_add_executable(convolution2d convolution2d.cu)
cuda_add_executable(histogram histogram.cu)

target_link_libraries(vectoradd ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(matrixmult ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CURAND_LIBRARIES})
# target_link_libraries(vectorsum ${CUDA_CUBLAS_LIBRARIES})

