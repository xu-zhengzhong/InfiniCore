#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/trmv.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_trmv(py::module &m) {
    m.def("trmv_",
          &op::trmv_,
          py::arg("a"),
          py::arg("x"),
          py::arg("uplo"),
          py::arg("trans"),
          py::arg("diag"),
          R"doc(In-place BLAS level-2 triangular matrix-vector multiply.)doc");
}

} // namespace infinicore::ops
