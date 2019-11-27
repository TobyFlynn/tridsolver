if (BUILD_TESTING)
  add_library(Catch INTERFACE)
  target_include_directories(Catch INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

  set(TEST_SOURCES
    catch_main.cpp)

  if (BUILD_FOR_CPU)
    option(BUILD_MPI_TESTS "Build the MPI tests" ON)
    set(TEST_SOURCES ${TEST_SOURCES} test_cpu.cpp)
  endif (BUILD_FOR_CPU)

  add_executable(tests ${TEST_SOURCES})
  target_link_libraries(tests Catch)

  if (BUILD_FOR_CPU)
    target_link_libraries(tests tridcpu)
    target_include_directories(tests PRIVATE ${PROJECT_SOURCE_DIR}/include ./)
  endif (BUILD_FOR_CPU)

  add_test(NAME catch_test
    COMMAND tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  if (BUILD_MPI_TESTS)
    set(MPI_TEST_NUM_PROC "2" CACHE STRING
      "The number of processes to launch in MPI tests.")
    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    add_executable(mpi_tests catch_mpi_main.cpp test_cpu_mpi.cpp)
    target_link_libraries(mpi_tests Catch)
    target_link_libraries(mpi_tests tridcpu)
    target_link_libraries(mpi_tests ${MPI_CXX_LIBRARIES})
    target_include_directories(mpi_tests PRIVATE ${PROJECT_SOURCE_DIR}/include ./)
    add_test(NAME catch_mpi_test
      COMMAND "mpirun" -np ${MPI_TEST_NUM_PROC} $<TARGET_FILE:mpi_tests>
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endif (BUILD_MPI_TESTS)

  if(BUILD_FOR_GPU)
    find_package(CUDA REQUIRED)
    include(FindCUDA)

    set(CUDA_64_BIT_DEVICE_CODE ON) # -m64 compiler option
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)

    set(CUDA_NVCC_FLAGS         "${CUDA_NVCC_FLAGS}         -arch sm_35 --ptxas-options=-v -Xptxas -dlcm=cg ")
    set(CUDA_NVCC_FLAGS_DEBUG   "${CUDA_NVCC_FLAGS_DEBUG}   -g -G -O0 -ftz=false -fmad=false                ")
    set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -O3                                             ")

    cuda_include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/test/)
    set(CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so)
    cuda_add_library(tridcuda_test_utils ${PROJECT_SOURCE_DIR}/test/test_cuda.cu SHARED OPTIONS -Xcompiler -fPIC) # Make binary relocatable
    cuda_add_cublas_to_target(tridcuda_test_utils)

    add_executable(cuda_tests catch_main.cpp)
    target_link_libraries(cuda_tests Catch)
    target_link_libraries(cuda_tests tridcuda tridcuda_test_utils)
    target_include_directories(cuda_tests PRIVATE ${PROJECT_SOURCE_DIR}/include ./)
    add_test(NAME catch_test_cuda
      COMMAND cuda_tests
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  endif (BUILD_FOR_GPU)
endif (BUILD_TESTING)
