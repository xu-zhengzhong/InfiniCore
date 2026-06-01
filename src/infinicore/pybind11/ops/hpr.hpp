#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/hpr.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_hpr(py::module &m) {
    m.def("hpr_",
          &op::hpr_,
          py::arg("alpha"),
          py::arg("x"),
          py::arg("ap"),
          py::arg("uplo"),
          R"doc(In-place BLAS level-2 Hermitian packed rank-1 update.)doc");
}

} // namespace infinicore::ops
