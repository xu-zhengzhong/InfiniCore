#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/gemm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_gemm(py::module &m) {
    m.def("gemm_",
          &op::gemm_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          py::arg("alpha"),
          py::arg("beta"),
          R"doc(In-place BLAS level-3 general matrix-matrix multiply.)doc");
}

} // namespace infinicore::ops
