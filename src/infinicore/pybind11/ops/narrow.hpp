#pragma once

#include <pybind11/pybind11.h>
#include "infinicore/ops/narrow.hpp" 

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_narrow(py::module &m) {
    m.def("narrow",
          &op::narrow,
          py::arg("input"),
          py::arg("dim"),
          py::arg("start"),
          py::arg("length"),
          R"doc(Returns a new tensor that is a narrowed version of input tensor.
The dimension dim is input from start to start + length.)doc");
    m.def("narrow_",
          &op::narrow_,
          py::arg("output"),
          py::arg("input"),
          py::arg("dim"),
          py::arg("start"),
          py::arg("length"),
          R"doc(Explicit output narrow operation. Writes the result into the output tensor.)doc");
}

} // namespace infinicore::ops
