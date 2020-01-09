#define CATCH_CONFIG_NOSTDOUT
#include "catch.hpp"
#include "catch_mpi_outputs.hpp"
#include "cuda_utils.hpp"
#include "cuda_mpi_wrappers.hpp"

#include <trid_common.h>
#include <trid_cuda.h>
#include <trid_mpi_cuda.hpp>
#include <trid_mpi_cpu.hpp>

#include "../src/cuda/trid_strided_multidim_mpi.hpp"
#include "../src/cuda/trid_linear_mpi.hpp"

#include <mpi.h>

#include <chrono>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

// Print routine for debugging
template <typename Container>
void print_array(const std::string &prompt, const Container &array) {
  Catch::cout() << prompt << ": [";
  for (size_t i = 0; i < array.size(); ++i) {
    Catch::cout() << (i == 0 ? "" : ", ") << std::setprecision(2) << array[i];
  }
  Catch::cout() << "]\n";
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

template <typename Float>
void require_allclose(const Float *expected, const Float *actual, size_t N,
                      int stride = 1, std::string value = "") {
  for (size_t j = 0, i = 0; j < N; ++j, i += stride) {
    CAPTURE(value);
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
  static const MPI_Datatype value; // = MPI_DOUBLE;
};
const MPI_Datatype ToMpiDatatype<double>::value = MPI_DOUBLE;

template <> struct ToMpiDatatype<float> {
  static const MPI_Datatype value; // = MPI_FLOAT;
};
const MPI_Datatype ToMpiDatatype<float>::value = MPI_FLOAT;

// Copies the local domain defined by `local_sizes` and `offsets` from the mesh.
//
// The 0th dimension is the contiguous one. The function is recursive; `dim` is
// current dimension, should equal one less than the number of dimensions when
// called from outside.
//
// `global_strides` is the product of the all global sizes in the lower
// dimensions (e.g. `global_strides[0] == 1`).
template <typename Float, unsigned Alignment>
void copy_strided(const AlignedArray<Float, Alignment> &src,
                  AlignedArray<Float, Alignment> &dest,
                  const std::vector<int> &local_sizes,
                  const std::vector<int> &offsets,
                  const std::vector<int> &global_strides, size_t dim,
                  int global_offset = 0) {
  if (dim == 0) {
    for (int i = 0; i < local_sizes[dim]; ++i) {
      dest.push_back(src[global_offset + offsets[dim] + i]);
    }
  } else {
    for (int i = 0; i < local_sizes[dim]; ++i) {
      const int new_global_offset =
          global_offset + (offsets[dim] + i) * global_strides[dim];
      copy_strided(src, dest, local_sizes, offsets, global_strides, dim - 1,
                   new_global_offset);
    }
  }
}

template <typename REAL>
void thomas_on_reduced_batched(const REAL *receive_buf, REAL *results,
                               int sys_n, int num_proc, int mpi_coord);

template <typename Float>
void test_manual_from_file(const std::string &file_name) {
  // The dimension of the MPI decomposition is the same as solve_dim
  MeshLoader<Float> mesh(file_name);

  int num_proc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Representation is column major: 0th dimension is the consecutive one
  const int eq_stride =
      std::accumulate(mesh.dims().data(), mesh.dims().data() + mesh.solve_dim(),
                      1, std::multiplies<int>());
  // The product of the sizes along the dimensions higher than solve_dim; needed
  // for the iteration later
  const int outer_size = std::accumulate(
      mesh.dims().data() + mesh.solve_dim() + 1,
      mesh.dims().data() + mesh.dims().size(), 1, std::multiplies<int>());
  const size_t eq_size = mesh.dims()[mesh.solve_dim()];
  // The number of systems to solve
  const int sys_n = eq_stride * outer_size;
  // The start index of our domain along the dimension of the MPI
  // decomposition/solve_dim
  const size_t mpi_domain_offset = rank * (eq_size / num_proc);
  // The size of the equations / our domain
  const size_t local_eq_size =
      rank == num_proc - 1 ? eq_size - mpi_domain_offset : eq_size / num_proc;

  /* Move data of the current rank to GPU */
  // Simulate distributed environment: only load our data
  const size_t domain_size = sys_n * local_eq_size;
  AlignedArray<Float, 1> a_host(domain_size), b_host(domain_size),
      c_host(domain_size), u_host(domain_size), d_host(domain_size);

  // sizes in each dimension for the current domain
  std::vector<int> local_sizes = mesh.dims();
  // MPI is only along one dimension
  local_sizes[mesh.solve_dim()] = local_eq_size;
  // offset of the beginning of the domain in each dimension
  // domain offset is 0 in each dimension except for solvedim
  std::vector<int> domain_offsets(mesh.dims().size());
  domain_offsets[mesh.solve_dim()] = mpi_domain_offset;
  // The strides in the mesh for each dimension.
  std::vector<int> global_strides(mesh.dims().size(), 1);
  std::partial_sum(mesh.dims().begin(),
                   mesh.dims().begin() + mesh.dims().size() - 1,
                   global_strides.begin() + 1, std::multiplies<int>{});

  copy_strided(mesh.a(), a_host, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.b(), b_host, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.c(), c_host, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.d(), d_host, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.u(), u_host, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);

  GPUMesh<Float> local_device_mesh(a_host, b_host, c_host, d_host, local_sizes);
  // arrays for local changes for each eq and buffer for boundary values
  DeviceArray<Float> aa_d(domain_size), cc_d(domain_size), dd_d(domain_size),
      boundaries(sys_n * 6);

  int blockdimx = 128; // Has to be the multiple of 4(or maybe 32??)
  int blockdimy = 1;
  int dimgrid = 1 + (sys_n - 1) / blockdimx; // can go up to 65535
  int dimgridx = dimgrid % 65536;            // can go up to max 65535 on Fermi
  int dimgridy = 1 + dimgrid / 65536;

  dim3 dimGrid_x(dimgridx, dimgridy);
  dim3 dimBlock_x(blockdimx, blockdimy);

  assert(local_eq_size > 2 &&
         "One of the processes has fewer than 2 equations, this is not "
         "supported\n");
  if (mesh.solve_dim() == 0) {
    trid_linear_forward<Float><<<dimGrid_x, dimBlock_x>>>(
        local_device_mesh.a().data(), local_device_mesh.b().data(),
        local_device_mesh.c().data(), local_device_mesh.d().data(), nullptr,
        aa_d.data(), cc_d.data(), dd_d.data(), boundaries.data(), local_eq_size,
        local_eq_size, sys_n);
  } else {
    DIM_V pads, dims;
    for (int i = 0; i < mesh.dims().size(); ++i) {
      pads.v[i] = local_sizes[i];
      dims.v[i] = local_sizes[i];
    }
    trid_strided_multidim_forward<Float><<<dimGrid_x, dimBlock_x>>>(
        local_device_mesh.a().data(), pads, local_device_mesh.b().data(), pads,
        local_device_mesh.c().data(), pads, local_device_mesh.d().data(), pads,
        nullptr, pads, aa_d.data(), cc_d.data(), dd_d.data(), boundaries.data(),
        mesh.dims().size(), mesh.solve_dim(), sys_n, dims);
  }

  AlignedArray<Float, 1> boundaries_res(sys_n * 6);
  boundaries_res.resize(sys_n * 6);
  cudaMemcpy(boundaries_res.data(), boundaries.data(),
             sizeof(Float) * 6 * sys_n, cudaMemcpyDeviceToHost);

  // ref
  std::vector<Float> aa_h(domain_size), cc_h(domain_size), dd_h(domain_size);
  cudaMemcpy(aa_h.data(), aa_d.data(), sizeof(Float) * domain_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(cc_h.data(), cc_d.data(), sizeof(Float) * domain_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(dd_h.data(), dd_d.data(), sizeof(Float) * domain_size,
             cudaMemcpyDeviceToHost);
  std::vector<Float> aa_ref(domain_size), cc_ref(domain_size), dd_ref(domain_size);

  std::vector<Float> aa(local_eq_size), cc(local_eq_size), dd(local_eq_size);
  AlignedArray<Float, 1> boundaries_ref(sys_n * 6);
  boundaries_ref.resize(sys_n * 6);
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

      boundaries_ref[(outer_ind * eq_stride + local_eq_start) * 6 + 0] = aa[0];
      boundaries_ref[(outer_ind * eq_stride + local_eq_start) * 6 + 1] =
          aa[local_eq_size - 1];
      boundaries_ref[(outer_ind * eq_stride + local_eq_start) * 6 + 2] = cc[0];
      boundaries_ref[(outer_ind * eq_stride + local_eq_start) * 6 + 3] =
          cc[local_eq_size - 1];
      boundaries_ref[(outer_ind * eq_stride + local_eq_start) * 6 + 4] = dd[0];
      boundaries_ref[(outer_ind * eq_stride + local_eq_start) * 6 + 5] =
          dd[local_eq_size - 1];
      for (int i = 0; i < aa.size(); ++i) {
        aa_ref[(outer_ind * local_eq_size * eq_stride + local_eq_start) +
               i * eq_stride] = aa[i];
        cc_ref[(outer_ind * local_eq_size * eq_stride + local_eq_start) +
               i * eq_stride] = cc[i];
        dd_ref[(outer_ind * local_eq_size * eq_stride + local_eq_start) +
               i * eq_stride] = dd[i];
      }
    }
  }
  // test the results of the forward run
  require_allclose(boundaries_ref, boundaries_res, sys_n * 6, 1);
  require_allclose(aa_ref.data(), aa_h.data(), domain_size, 1, "aa");
  require_allclose(cc_ref.data(), cc_h.data(), domain_size, 1, "cc");
  require_allclose(dd_ref.data(), dd_h.data(), domain_size, 1, "dd");
  // test thomas on reduced
  const int comm_buff_size = 6 * sys_n;
  std::vector<Float> receive_buf(comm_buff_size * num_proc);
  MPI_Allgather(boundaries_res.data(), comm_buff_size,
                ToMpiDatatype<Float>::value, receive_buf.data(), comm_buff_size,
                ToMpiDatatype<Float>::value, MPI_COMM_WORLD);
  thomas_on_reduced_batched(receive_buf.data(), boundaries_res.data(), sys_n,
                            num_proc, rank);

  cudaMemcpy(boundaries.data(), boundaries_res.data(), sizeof(Float) * 2 * sys_n,
             cudaMemcpyHostToDevice);
  if (mesh.solve_dim() == 0) {
    trid_linear_backward<Float><<<dimGrid_x, dimBlock_x>>>(
        aa_d.data(), cc_d.data(), dd_d.data(), local_device_mesh.d().data(),
        nullptr, boundaries.data(), local_eq_size, local_eq_size, sys_n);
  } else {
    DIM_V pads, dims;
    for (int i = 0; i < mesh.dims().size(); ++i) {
      pads.v[i] = local_sizes[i];
      dims.v[i] = local_sizes[i];
    }
    trid_strided_multidim_backward<Float><<<dimGrid_x, dimBlock_x>>>(
        aa_d.data(), pads, cc_d.data(), pads, dd_d.data(),
        local_device_mesh.d().data(), pads, nullptr, pads, boundaries.data(),
        mesh.dims().size(), mesh.solve_dim(), sys_n, dims);
  }

  cudaMemcpy(d_host.data(), local_device_mesh.d().data(), sizeof(Float) * domain_size,
             cudaMemcpyDeviceToHost);
  require_allclose(u_host.data(), d_host.data(), domain_size, 1, "result");
}

template <typename Float>
void test_solver_from_file(const std::string &file_name) {
  // The dimension of the MPI decomposition is the same as solve_dim
  MeshLoader<Float> mesh(file_name);

  int num_proc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Create rectangular grid
  std::vector<int> mpi_dims(mesh.dims().size(), 0);
  MPI_Dims_create(num_proc, mesh.dims().size(), mpi_dims.data());

  // Create communicator for grid
  MPI_Comm cart_comm;
  std::vector<int> periods(mesh.dims().size(), 0);

  MPI_Cart_create(MPI_COMM_WORLD, mesh.dims().size(), mpi_dims.data(),
                  periods.data(), 0, &cart_comm);

  MpiSolverParams params(cart_comm, mesh.dims().size(), mpi_dims.data());

  // The size of the local domain.
  std::vector<int> local_sizes(mesh.dims().size());
  // The starting indices of the local domain in each dimension.
  std::vector<int> domain_offsets(mesh.dims().size());
  // The strides in the mesh for each dimension.
  std::vector<int> global_strides(mesh.dims().size());
  int domain_size = 1;
  for (size_t i = 0; i < local_sizes.size(); ++i) {
    const int global_dim = mesh.dims()[i];
    domain_offsets[i] = params.mpi_coords[i] * (global_dim / mpi_dims[i]);
    local_sizes[i] = params.mpi_coords[i] == mpi_dims[i] - 1
                         ? global_dim - domain_offsets[i]
                         : global_dim / mpi_dims[i];
    global_strides[i] = i == 0 ? 1 : global_strides[i - 1] * mesh.dims()[i - 1];
    domain_size *= local_sizes[i];
  }

  // Simulate distributed environment: only load our data
  AlignedArray<Float, 1> a(domain_size), b(domain_size), c(domain_size),
      u(domain_size), d(domain_size);
  copy_strided(mesh.a(), a, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.b(), b, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.c(), c, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.d(), d, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);
  copy_strided(mesh.u(), u, local_sizes, domain_offsets, global_strides,
               local_sizes.size() - 1);

  GPUMesh<Float> local_device_mesh(a, b, c, d, local_sizes);

  // Solve the equations
  tridmtsvStridedBatchMPIWrapper<Float>(
      params, local_device_mesh.a().data(), local_device_mesh.b().data(),
      local_device_mesh.c().data(), local_device_mesh.d().data(), nullptr,
      local_sizes.data(), local_sizes.size(), mesh.solve_dim(),
      local_sizes.data());

  // Check result
  cudaMemcpy(d.data(), local_device_mesh.d().data(),
             sizeof(Float) * domain_size, cudaMemcpyDeviceToHost);

  require_allclose(u, d, domain_size, 1);
}

TEST_CASE("cuda manual mpi: solveX", "[manual][solvedim:0]") {
  SECTION("double") {
    SECTION("ndims: 1") {
      test_manual_from_file<double>("files/one_dim_large");
    }
    SECTION("ndims: 2") {
      test_manual_from_file<double>("files/two_dim_large_solve0");
    }
    SECTION("ndims: 3") {
      test_manual_from_file<double>("files/three_dim_large_solve0");
    }
    SECTION("ndims: 4") {
      test_manual_from_file<double>("files/four_dim_large_solve0");
    }
  }
  SECTION("float") {
    SECTION("ndims: 1") {
      test_manual_from_file<float>("files/one_dim_large");
    }
    SECTION("ndims: 2") {
      test_manual_from_file<float>("files/two_dim_large_solve0");
    }
    SECTION("ndims: 3") {
      test_manual_from_file<float>("files/three_dim_large_solve0");
    }
    SECTION("ndims: 4") {
      test_manual_from_file<float>("files/four_dim_large_solve0");
    }
  }
}

TEST_CASE("cuda manual mpi: solveY", "[manual][solvedim:1]") {
  SECTION("double") {
    SECTION("ndims: 2") {
      test_manual_from_file<double>("files/two_dim_large_solve1");
    }
    SECTION("ndims: 3") {
      test_manual_from_file<double>("files/three_dim_large_solve1");
    }
    SECTION("ndims: 4") {
      test_manual_from_file<double>("files/four_dim_large_solve1");
    }
  }
  SECTION("float") {
    SECTION("ndims: 2") {
      test_manual_from_file<float>("files/two_dim_large_solve1");
    }
    SECTION("ndims: 3") {
      test_manual_from_file<float>("files/three_dim_large_solve1");
    }
    SECTION("ndims: 4") {
      test_manual_from_file<float>("files/four_dim_large_solve1");
    }
  }
}

TEST_CASE("cuda manual mpi: solveZ", "[manual][solvedim:2]") {
  SECTION("double") {
    SECTION("ndims: 3") {
      test_manual_from_file<double>("files/three_dim_large_solve2");
    }
    SECTION("ndims: 4") {
      test_manual_from_file<double>("files/four_dim_large_solve2");
    }
  }
  SECTION("float") {
    SECTION("ndims: 3") {
      test_manual_from_file<float>("files/three_dim_large_solve2");
    }
    SECTION("ndims: 4") {
      test_manual_from_file<float>("files/four_dim_large_solve2");
    }
  }
}

TEST_CASE("cuda manual mpi: solve3", "[manual][solvedim:3]") {
  SECTION("double") {
    SECTION("ndims: 4") {
      test_manual_from_file<double>("files/four_dim_large_solve3");
    }
  }
  SECTION("float") {
    SECTION("ndims: 4") {
      test_manual_from_file<float>("files/four_dim_large_solve3");
    }
  }
}

TEST_CASE("cuda solver mpi: solveX", "[solver][solvedim:0]") {
  SECTION("double") {
    SECTION("ndims: 1") {
      test_solver_from_file<double>("files/one_dim_large");
    }
    SECTION("ndims: 2") {
      test_solver_from_file<double>("files/two_dim_large_solve0");
    }
    SECTION("ndims: 3") {
      test_solver_from_file<double>("files/three_dim_large_solve0");
    }
    SECTION("ndims: 4") {
      test_solver_from_file<double>("files/four_dim_large_solve0");
    }
  }
  SECTION("float") {
    SECTION("ndims: 1") { test_solver_from_file<float>("files/one_dim_large"); }
    SECTION("ndims: 2") {
      test_solver_from_file<float>("files/two_dim_large_solve0");
    }
    SECTION("ndims: 3") {
      test_solver_from_file<float>("files/three_dim_large_solve0");
    }
    SECTION("ndims: 4") {
      test_solver_from_file<float>("files/four_dim_large_solve0");
    }
  }
}

TEST_CASE("cuda solver mpi: solveY", "[solver][solvedim:1]") {
  SECTION("double") {
    SECTION("ndims: 2") {
      test_solver_from_file<double>("files/two_dim_large_solve1");
    }
    SECTION("ndims: 3") {
      test_solver_from_file<double>("files/three_dim_large_solve1");
    }
    SECTION("ndims: 4") {
      test_solver_from_file<double>("files/four_dim_large_solve1");
    }
  }
  SECTION("float") {
    SECTION("ndims: 2") {
      test_solver_from_file<float>("files/two_dim_large_solve1");
    }
    SECTION("ndims: 3") {
      test_solver_from_file<float>("files/three_dim_large_solve1");
    }
    SECTION("ndims: 4") {
      test_solver_from_file<float>("files/four_dim_large_solve1");
    }
  }
}

TEST_CASE("cuda solver mpi: solveZ", "[solver][solvedim:2]") {
  SECTION("double") {
    SECTION("ndims: 3") {
      test_solver_from_file<double>("files/three_dim_large_solve2");
    }
    SECTION("ndims: 4") {
      test_solver_from_file<double>("files/four_dim_large_solve2");
    }
  }
  SECTION("float") {
    SECTION("ndims: 3") {
      test_solver_from_file<float>("files/three_dim_large_solve2");
    }
    SECTION("ndims: 4") {
      test_solver_from_file<float>("files/four_dim_large_solve2");
    }
  }
}

TEST_CASE("cuda solver mpi: solve3", "[solver][solvedim:3]") {
  SECTION("double") {
    SECTION("ndims: 4") {
      test_solver_from_file<double>("files/four_dim_large_solve3");
    }
  }
  SECTION("float") {
    SECTION("ndims: 4") {
      test_solver_from_file<float>("files/four_dim_large_solve3");
    }
  }
}
