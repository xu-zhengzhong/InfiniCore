#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/rotmg.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_rotmg(py::module &m) {
    m.def("rotmg",
          &op::rotmg,
          py::arg("d1"),
          py::arg("d2"),
          py::arg("x1"),
          py::arg("y1"),
          R"doc(BLAS Level1 rotmg function.)doc");

    m.def("rotmg_",
          &op::rotmg_,
          py::arg("d1"),
          py::arg("d2"),
          py::arg("x1"),
          py::arg("y1"),
          py::arg("out_d1"),
          py::arg("out_d2"),
          py::arg("out_x1"),
          py::arg("out_param"),
          R"doc(BLAS Level1 in-place rotmg function.)doc");
}

} // namespace infinicore::ops