#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/her2.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_her2(py::module &m) {
    m.def("her2_",
          &op::her2_,
          py::arg("alpha"),
          py::arg("x"),
          py::arg("y"),
          py::arg("a"),
          py::arg("uplo"),
          R"doc(In-place BLAS level-2 Hermitian rank-2 update.)doc");
}

} // namespace infinicore::ops
