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
    add_executable(mpi_tests catch_main.cpp test_cpu_mpi.cpp)
    target_link_libraries(mpi_tests Catch)
    target_link_libraries(tests tridcpu)
    target_include_directories(tests PRIVATE ${PROJECT_SOURCE_DIR}/include ./)
    add_test(NAME catch_mpi_test
      COMMAND "mpirun -np ${MPI_TEST_NUM_PROC} $<TARGET_FILE:mpi_tests>"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endif (BUILD_MPI_TESTS)
endif (BUILD_TESTING)
