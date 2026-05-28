#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/tbmv.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_tbmv(py::module &m) {
    m.def("tbmv_",
          &op::tbmv_,
          py::arg("a"),
          py::arg("x"),
          py::arg("uplo"),
          py::arg("trans"),
          py::arg("diag"),
          py::arg("k"),
          R"doc(In-place BLAS level-2 triangular band matrix-vector multiply.)doc");
}

} // namespace infinicore::ops
