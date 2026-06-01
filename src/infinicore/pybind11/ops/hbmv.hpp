#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/hbmv.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_hbmv(py::module &m) {
    m.def("hbmv_",
          &op::hbmv_,
          py::arg("alpha"),
          py::arg("a"),
          py::arg("x"),
          py::arg("beta"),
          py::arg("y"),
          py::arg("uplo"),
          py::arg("k"),
          R"doc(In-place BLAS level-2 Hermitian band matrix-vector multiply.)doc");
}

} // namespace infinicore::ops
