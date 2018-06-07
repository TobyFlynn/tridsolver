#include "catch.hpp"
#include "utils.hpp"

#include <trid_cpu.h>
#include <trid_mpi_cpu.hpp>

#include <mpi.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <thread>

// Some routines for debugging
void random_wait() {
  std::this_thread::sleep_for(std::chrono::milliseconds(std::rand() % 500));
}

template <typename Container>
void print_array(const std::string &prompt,
                 const Container &array) {
  std::stringstream ss;
  ss << prompt << ": [";
  for (size_t i = 0; i < array.size(); ++i) {
    ss << (i == 0 ? "" : ", ") << std::setprecision(10) << array[i];
  }
  ss << "]";
  random_wait();
  std::cout << ss.str() << std::endl << std::flush;
}

template <typename Float, unsigned Align>
void require_allclose(const AlignedArray<Float, Align> &expected,
                      const AlignedArray<Float, Align> &actual, size_t N = 0,
                      int stride = 1) {
  if (N == 0) {
    assert(expected.size() == actual.size());
    N = expected.size();
  }
  for (size_t j = 0, i = 0; j < N; ++j, i += stride) {
    CAPTURE(i);
    CAPTURE(expected[i]);
    CAPTURE(actual[i]);
    Float min_val = std::min(std::abs(expected[i]), std::abs(actual[i]));
    const double abs_tolerance =
        std::is_same<Float, float>::value ? ABS_TOLERANCE_FLOAT : ABS_TOLERANCE;
    const double rel_tolerance =
        std::is_same<Float, float>::value ? REL_TOLERANCE_FLOAT : REL_TOLERANCE;
    const double tolerance = abs_tolerance + rel_tolerance * min_val;
    CAPTURE(tolerance);
    const double diff = std::abs(static_cast<double>(expected[i]) - actual[i]);
    CAPTURE(diff);
    REQUIRE(diff <= tolerance);
  }
}

template <typename Float> struct ToMpiDatatype {};

template <> struct ToMpiDatatype<double> {
  static const MPI_Datatype value = MPI_DOUBLE;
};

template <> struct ToMpiDatatype<float> {
  static const MPI_Datatype value = MPI_FLOAT;
};

template <typename Float> void test_from_file(const std::string &file_name) {
  MeshLoader<Float> mesh(file_name);
  assert(mesh.dims().size() == 1 && "Only one dimension is supported");

  int num_proc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int stride = 1;
  /* for (size_t i = 0; i < mesh.solve_dim(); ++i) { */
  /*   stride *= mesh.dims()[i]; */
  /* } */
  const size_t global_N = mesh.dims()[mesh.solve_dim()];
  const size_t offset = rank * (global_N / num_proc);
  const size_t local_N =
      rank == num_proc - 1 ? global_N - offset : global_N / num_proc;
  std::vector<Float> aa(local_N), cc(local_N), dd(local_N);
  AlignedArray<Float, 1> d(mesh.d(), offset, offset + local_N),
      u(mesh.u(), offset, offset + local_N);

  // TODO in input, how is the indexing (starting from 0/1/etc)
  thomas_forward<Float>(mesh.a().data() + offset, mesh.b().data() + offset,
                        mesh.c().data() + offset, d.data(), nullptr,
                        aa.data(), cc.data(), dd.data(), local_N, stride);

  std::vector<Float> send_buf(6), receive_buf(6 * num_proc);
  send_buf[0] = aa[0];
  send_buf[1] = aa[local_N - 1];
  send_buf[2] = cc[0];
  send_buf[3] = cc[local_N - 1];
  send_buf[4] = dd[0];
  send_buf[5] = dd[local_N - 1];
  MPI_Allgather(send_buf.data(), 6, ToMpiDatatype<Float>::value,
                receive_buf.data(), 6, ToMpiDatatype<Float>::value,
                MPI_COMM_WORLD);
  // Reduced system
  std::vector<Float> aa_r(2 * num_proc), cc_r(2 * num_proc), dd_r(2 * num_proc);
  for (int i = 0; i < num_proc; ++i) {
    aa_r[2 * i + 0] = receive_buf[6 * i + 0];
    aa_r[2 * i + 1] = receive_buf[6 * i + 1];
    cc_r[2 * i + 0] = receive_buf[6 * i + 2];
    cc_r[2 * i + 1] = receive_buf[6 * i + 3];
    dd_r[2 * i + 0] = receive_buf[6 * i + 4];
    dd_r[2 * i + 1] = receive_buf[6 * i + 5];
  }
  // indexing of cc_r, dd_r starts from 0
  // while indexing of aa_r starts from 1
  thomas_on_reduced(aa_r.data(), cc_r.data(), dd_r.data(), 2 * num_proc,
                    stride);

  dd[0] = dd_r[2 * rank]; // TODO check this
  dd[local_N - 1] = dd_r[2 * rank + 1];
  thomas_backward(aa.data(), cc.data(), dd.data(), d.data(), local_N, stride);

  require_allclose(u, d, local_N, 1);
}

TEST_CASE("mpi: small") {
  SECTION("double") {
    SECTION("ndims: 1") { test_from_file<double>("files/one_dim_small"); }
  }
  SECTION("float") {
    SECTION("ndims: 1") { test_from_file<double>("files/one_dim_small"); }
  }
}
