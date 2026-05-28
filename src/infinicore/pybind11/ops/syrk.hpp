#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/syrk.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_syrk(py::module &m) {
    m.def("syrk_",
          &op::syrk_,
          py::arg("a"),
          py::arg("alpha"),
          py::arg("beta"),
          py::arg("c"),
          py::arg("uplo"),
          py::arg("trans"),
          R"doc(In-place BLAS level-3 symmetric rank-k update.)doc");
}

} // namespace infinicore::ops
