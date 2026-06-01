#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/hpr2.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_hpr2(py::module &m) {
    m.def("hpr2_",
          &op::hpr2_,
          py::arg("alpha"),
          py::arg("x"),
          py::arg("y"),
          py::arg("ap"),
          py::arg("uplo"),
          R"doc(In-place BLAS level-2 Hermitian packed rank-2 update.)doc");
}

} // namespace infinicore::ops
