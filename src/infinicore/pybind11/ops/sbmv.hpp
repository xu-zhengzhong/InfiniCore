#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/sbmv.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_sbmv(py::module &m) {
    m.def("sbmv_",
          &op::sbmv_,
          py::arg("alpha"),
          py::arg("a"),
          py::arg("x"),
          py::arg("beta"),
          py::arg("y"),
          py::arg("uplo"),
          py::arg("k"),
          R"doc(In-place BLAS level-2 symmetric band matrix-vector multiply.)doc");
}

} // namespace infinicore::ops
