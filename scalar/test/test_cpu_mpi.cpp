#include "catch.hpp"
#include "utils.hpp"

#include <trid_cpu.h>
#include <trid_mpi_cpu.hpp>

#include <mpi.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

// Some routines for debugging
void random_wait() {
  std::this_thread::sleep_for(std::chrono::milliseconds(std::rand() % 500));
}

template <typename Container>
void print_array(const std::string &prompt, const Container &array) {
  std::stringstream ss;
  ss << prompt << ": [";
  for (size_t i = 0; i < array.size(); ++i) {
    ss << (i == 0 ? "" : ", ") << std::setprecision(2) << array[i];
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
  // The dimension of the MPI decomposition is the same as solve_dim
  MeshLoader<Float> mesh(file_name);

  int num_proc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Representation is row major: 0th dimension is the consecutive one
  int eq_stride = 1;
  for (size_t i = 0; i < mesh.solve_dim(); ++i) {
    eq_stride *= mesh.dims()[i];
  }
  // The product of the sizes along the dimensions higher than solve_dim; needed
  // for the iteration later
  int outer_size = 1;
  for (size_t i = mesh.solve_dim() + 1; i < mesh.dims().size(); ++i) {
    outer_size *= mesh.dims()[i];
  }
  const size_t eq_size = mesh.dims()[mesh.solve_dim()];
  // The start index of our domain along the dimension of the MPI
  // decomposition/solve_dim
  const size_t mpi_domain_offset = rank * (eq_size / num_proc);
  // The size of the equations / our domain
  const size_t local_eq_size =
      rank == num_proc - 1 ? eq_size - mpi_domain_offset : eq_size / num_proc;

  // Local modifications to the coefficients
  std::vector<Float> aa(local_eq_size), cc(local_eq_size), dd(local_eq_size);
  // MPI buffers
  std::vector<Float> send_buf(6), receive_buf(6 * num_proc);
  // Reduced system
  std::vector<Float> aa_r(2 * num_proc), cc_r(2 * num_proc), dd_r(2 * num_proc);

  for (size_t outer_ind = 0; outer_ind < outer_size; ++outer_ind) {
    const size_t domain_start =
        outer_ind * eq_size * eq_stride + mpi_domain_offset * eq_stride;
    const size_t domain_size = local_eq_size * eq_stride;
    // Simulate distributed environment: only load our data
    const AlignedArray<Float, 1> a(mesh.a(), domain_start,
                                   domain_start + domain_size),
        b(mesh.b(), domain_start, domain_start + domain_size),
        c(mesh.c(), domain_start, domain_start + domain_size),
        u(mesh.u(), domain_start, domain_start + domain_size);
    AlignedArray<Float, 1> d(mesh.d(), domain_start,
                             domain_start + domain_size);

    for (size_t local_eq_start = 0; local_eq_start < eq_stride;
         ++local_eq_start) {

      thomas_forward<Float>(
          a.data() + local_eq_start, b.data() + local_eq_start,
          c.data() + local_eq_start, d.data() + local_eq_start, nullptr,
          aa.data(), cc.data(), dd.data(), local_eq_size, eq_stride);

      send_buf[0] = aa[0];
      send_buf[1] = aa[local_eq_size - 1];
      send_buf[2] = cc[0];
      send_buf[3] = cc[local_eq_size - 1];
      send_buf[4] = dd[0];
      send_buf[5] = dd[local_eq_size - 1];
      MPI_Allgather(send_buf.data(), 6, ToMpiDatatype<Float>::value,
                    receive_buf.data(), 6, ToMpiDatatype<Float>::value,
                    MPI_COMM_WORLD);
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
      thomas_on_reduced(aa_r.data(), cc_r.data(), dd_r.data(), 2 * num_proc, 1);

      dd[0] = dd_r[2 * rank];
      dd[local_eq_size - 1] = dd_r[2 * rank + 1];
      thomas_backward(aa.data(), cc.data(), dd.data(),
                      d.data() + local_eq_start, local_eq_size, eq_stride);

    }
    require_allclose(u, d, domain_size, 1);
  }
}

TEST_CASE("mpi: small") {
  SECTION("double") {
    SECTION("ndims: 1") { test_from_file<double>("files/one_dim_small"); }
    SECTION("ndims: 2") {
      SECTION("solvedim: 0") {
        test_from_file<double>("files/two_dim_small_solve0");
      }
      SECTION("solvedim: 1") {
        test_from_file<double>("files/two_dim_small_solve1");
      }
    }
  }
  SECTION("float") {
    SECTION("ndims: 1") { test_from_file<float>("files/one_dim_small"); }
    SECTION("ndims: 2") {
      SECTION("solvedim: 0") {
        test_from_file<float>("files/two_dim_small_solve0");
      }
      SECTION("solvedim: 1") {
        test_from_file<float>("files/two_dim_small_solve1");
      }
    }
  }
}

TEST_CASE("mpi: large", "[large]") {
  SECTION("double") {
    SECTION("ndims: 1") { test_from_file<double>("files/one_dim_large"); }
    SECTION("ndims: 2") {
      SECTION("solvedim: 0") {
        test_from_file<double>("files/two_dim_large_solve0");
      }
      SECTION("solvedim: 1") {
        test_from_file<double>("files/two_dim_large_solve1");
      }
    }
  }
  SECTION("float") {
    SECTION("ndims: 1") { test_from_file<float>("files/one_dim_large"); }
    SECTION("ndims: 2") {
      SECTION("solvedim: 0") {
        test_from_file<float>("files/two_dim_large_solve0");
      }
      SECTION("solvedim: 1") {
        test_from_file<float>("files/two_dim_large_solve1");
      }
    }
  }
}
