#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/her2k.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_her2k(py::module &m) {
    m.def("her2k_",
          &op::her2k_,
          py::arg("a"),
          py::arg("b"),
          py::arg("alpha"),
          py::arg("beta"),
          py::arg("c"),
          py::arg("uplo"),
          py::arg("trans"),
          R"doc(In-place BLAS level-3 Hermitian rank-2k update.)doc");
}

} // namespace infinicore::ops
