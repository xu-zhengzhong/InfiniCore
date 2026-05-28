#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/gemv.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_gemv(py::module &m) {
    m.def("gemv_",
          &op::gemv_,
          py::arg("alpha"),
          py::arg("a"),
          py::arg("x"),
          py::arg("beta"),
          py::arg("y"),
          py::arg("trans"),
          R"doc(In-place BLAS level-2 matrix-vector multiply.)doc");
}

} // namespace infinicore::ops
