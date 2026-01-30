#pragma once

#include "infinicore/ops/sgn.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_sgn(py::module &m) {
    m.def("sgn",
          &op::sgn,
          py::arg("input"),
          R"doc(Computes the sgn (sign with normalized magnitude) of the input tensor.
          
This function is an extension of sign() to complex tensors. It computes a 
new tensor whose elements have the same angles as the corresponding elements 
of input and absolute values (i.e. magnitudes) of one for complex tensors 
and is equivalent to sign() for non-complex tensors.)doc");

    m.def("sgn_",
          &op::sgn_,
          py::arg("input"),
          py::arg("output"),
          R"doc(In-place sgn.)doc");
}

} // namespace infinicore::ops