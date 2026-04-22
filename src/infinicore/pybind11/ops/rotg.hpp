#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/rotg.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_rotg(py::module &m) {
    m.def("rotg_",
          &op::rotg_,
          py::arg("a"),
          py::arg("b"),
          py::arg("c"),
          py::arg("s"),
          R"doc(BLAS Level1 in-place rotg function.)doc");
}

} // namespace infinicore::ops