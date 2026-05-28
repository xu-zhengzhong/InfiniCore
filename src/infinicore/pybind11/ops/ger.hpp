#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/ger.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_ger(py::module &m) {
    m.def("ger_",
          &op::ger_,
          py::arg("alpha"),
          py::arg("x"),
          py::arg("y"),
          py::arg("a"),
          R"doc(In-place BLAS level-2 general rank-1 update.)doc");
}

} // namespace infinicore::ops
