#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/symv.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_symv(py::module &m) {
    m.def("symv_",
          &op::symv_,
          py::arg("alpha"),
          py::arg("a"),
          py::arg("x"),
          py::arg("beta"),
          py::arg("y"),
          py::arg("uplo"),
          R"doc(In-place BLAS level-2 symmetric matrix-vector multiply.)doc");
}

} // namespace infinicore::ops
