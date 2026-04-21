#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/rotg.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_rotg(py::module &m) {
    m.def("rotg",
          &op::rotg,
          py::arg("a"),
          py::arg("b"),
          R"doc(BLAS Level1 rotg function.)doc");

    m.def("rotg_",
          &op::rotg_,
          py::arg("a"),
          py::arg("b"),
          py::arg("out_a"),
          py::arg("out_b"),
          py::arg("out_c"),
          py::arg("out_s"),
          R"doc(BLAS Level1 in-place rotg function.)doc");
}

} // namespace infinicore::ops