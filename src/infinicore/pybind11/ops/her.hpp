#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/her.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_her(py::module &m) {
    m.def("her_",
          &op::her_,
          py::arg("alpha"),
          py::arg("x"),
          py::arg("a"),
          py::arg("uplo"),
          R"doc(In-place BLAS level-2 Hermitian rank-1 update.)doc");
}

} // namespace infinicore::ops
