#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/trsm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_trsm(py::module &m) {
    m.def("trsm_",
          &op::trsm_,
          py::arg("a"),
          py::arg("alpha"),
          py::arg("b"),
          py::arg("side"),
          py::arg("uplo"),
          py::arg("trans"),
          py::arg("diag"),
          R"doc(In-place BLAS level-3 triangular solve with multiple right-hand sides.)doc");
}

} // namespace infinicore::ops
