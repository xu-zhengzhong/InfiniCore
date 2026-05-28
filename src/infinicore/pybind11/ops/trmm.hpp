#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/trmm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_trmm(py::module &m) {
    m.def("trmm_",
          &op::trmm_,
          py::arg("a"),
          py::arg("alpha"),
          py::arg("b"),
          py::arg("side"),
          py::arg("uplo"),
          py::arg("trans"),
          py::arg("diag"),
          R"doc(In-place BLAS level-3 triangular matrix-matrix multiply.)doc");
}

} // namespace infinicore::ops
