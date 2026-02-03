#pragma once

#include "infinicore/ops/rot90.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_rot90(py::module &m) {
    m.def("rot90",
          &op::rot90,
          py::arg("input"),
          py::arg("k") = 1,
          py::arg("dims") = std::vector<int64_t>{0, 1},
          R"doc(Rotate an n-D tensor by 90 degrees in the plane specified by dims axis.
          
Rotation direction is from the first towards the second axis if k > 0, 
and from the second towards the first for k < 0.

Parameters:
    input (Tensor): the input tensor.
    k (int): number of times to rotate. Default value is 1.
    dims (list): axis to rotate. Default value is [0, 1].)doc");

    m.def("rot90_",
          &op::rot90_,
          py::arg("input"),
          py::arg("output"),
          py::arg("k") = 1,
          py::arg("dims") = std::vector<int64_t>{0, 1},
          R"doc(In-place rot90.)doc");
}

} // namespace infinicore::ops