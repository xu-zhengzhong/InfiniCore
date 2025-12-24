#pragma once

#include <pybind11/pybind11.h>
#include "infinicore/ops/log.hpp" // 引用核心算子头文件

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_log(py::module &m) {
    // 绑定 out-of-place 接口: output = log(input)
    m.def("log",
          &op::log,
          py::arg("input"),
          R"doc(Computes the natural logarithm of each element of input.
          
Returns a new tensor with the natural logarithm of the elements of input.)doc");

    // 绑定 in-place 接口: log_(output, input)
    m.def("log_",
          &op::log_,
          py::arg("output"),
          py::arg("input"),
          R"doc(In-place log operation. Writes result into output tensor.)doc");
}

} // namespace infinicore::ops
