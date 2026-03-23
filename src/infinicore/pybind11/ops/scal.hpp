#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/scal.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_scal(py::module &m) {
    m.def("scal",
          &op::scal,
          py::arg("x"),
          py::arg("alpha"),
          R"doc(Element-wise scaling function.)doc");

    m.def("scal_",
          &op::scal_,
          py::arg("y"),
          py::arg("x"),
          py::arg("alpha"),
          R"doc(In-place element-wise scaling function.)doc");
}

} // namespace infinicore::ops
