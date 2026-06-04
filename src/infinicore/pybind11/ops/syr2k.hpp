#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/syr2k.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_syr2k(py::module &m) {
    m.def("syr2k_",
          &op::syr2k_,
          py::arg("a"),
          py::arg("b"),
          py::arg("alpha"),
          py::arg("beta"),
          py::arg("c"),
          py::arg("uplo"),
          py::arg("trans"),
          R"doc(In-place BLAS level-3 symmetric rank-2k update.)doc");
}

} // namespace infinicore::ops
