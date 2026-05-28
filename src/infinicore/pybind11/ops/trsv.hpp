#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/trsv.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_trsv(py::module &m) {
    m.def("trsv_",
          &op::trsv_,
          py::arg("a"),
          py::arg("x"),
          py::arg("uplo"),
          py::arg("trans"),
          py::arg("diag"),
          R"doc(In-place BLAS level-2 triangular solve.)doc");
}

} // namespace infinicore::ops
