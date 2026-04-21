#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/rotm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_rotm(py::module &m) {
    m.def("rotm",
          &op::rotm,
          py::arg("x"),
          py::arg("y"),
          py::arg("param"),
          R"doc(BLAS Level1 rotm function.)doc");

    m.def("rotm_",
          &op::rotm_,
          py::arg("x"),
          py::arg("y"),
          py::arg("param"),
          py::arg("out_x"),
          py::arg("out_y"),
          R"doc(BLAS Level1 in-place rotm function.)doc");
}

} // namespace infinicore::ops