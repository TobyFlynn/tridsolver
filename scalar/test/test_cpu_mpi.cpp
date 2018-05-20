#include "catch.hpp"
#include "utils.hpp"

#include <trid_cpu.h>
#include <trid_mpi_cpu.h>

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

template <typename Float> void test_from_file(const std::string &file_name) {
  MeshLoader<Float> mesh(file_name);
  AlignedArray<Float, 1> d(mesh.d());
  assert(mesh.dims.size() == 1 && "Only one dimension is supported");

  int num_proc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int stride = 1;
  /* for (size_t i = 0; i < mesh.solve_dim(); ++i) { */
  /*   stride *= mesh.dims()[i]; */
  /* } */
  const size_t global_N = mesh.dims()[mesh.solve_dim()];
  const size_t offset = rank * (global_N / num_proc);
  const size_t local_N = std::min(global_N / num_proc, global_N - offset);
  std::vector<Float> aa(local_N), cc(local_N), dd(local_N);

  // TODO in input, how is the indexing (starting from 0/1/etc)
  thomas_forward(mesh.a().data() + offset, mesh.b().data() + offset,
                 mesh.c().data() + offset, d().data() + offset, nullptr,
                 aa.data(), cc.data(), dd.data(), local_N, stride);

  std::vector<Float> send_buf(6), receive_buf();
  send_buf[0] = aa[0];
  send_buf[1] = aa[local_N - 1];
  send_buf[2] = bb[0];
  send_buf[3] = bb[local_N - 1];
  send_buf[4] = cc[0];
  send_buf[5] = cc[local_N - 1];
  // TODO allgather
  // Reduced system
  std::vector<Float> aa_r(2 * num_proc), cc_r(2 * num_proc), dd_r(2 * num_proc);
  // TODO extract aa_r, etc.
  thomas_on_reduced(aa_r.data(), cc_r.data(), dd_r.data(), 2 * num_proc,
                    stride);

  dd[0] = dd_r[2 * rank]; // TODO check this
  dd[local_N - 1] = dd_r[2 * rank + 1];
  thomas_backward(aa.data(), cc.data(), dd.data(), d.data(), local_N, stride);

  require_allclose(mesh.u(), d, global_N, 1);
}

TEST_CASE("mpi: small") {
  SECTION("double") {
    SECTION("ndims: 1") { test_from_file<double>("files/one_dim_small"); }
  }
  SECTION("float") {
    SECTION("ndims: 1") { test_from_file<double>("files/one_dim_small"); }
  }
}
