#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/herk.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_herk(py::module &m) {
    m.def("herk_",
          &op::herk_,
          py::arg("a"),
          py::arg("alpha"),
          py::arg("beta"),
          py::arg("c"),
          py::arg("uplo"),
          py::arg("trans"),
          R"doc(In-place BLAS level-3 Hermitian rank-k update.)doc");
}

} // namespace infinicore::ops
