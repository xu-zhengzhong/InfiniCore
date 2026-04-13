#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/scal.hpp"
#include "../../../utils/custom_types.h"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_scal(py::module &m) {
      m.def(
            "scal",
            [](Tensor x, double alpha) {
                  switch (x->dtype()) {
                  case DataType::F16: {
                        fp16_t a = utils::cast<fp16_t>(static_cast<float>(alpha));
                        return op::scal(x, &a);
                  }
                  case DataType::BF16: {
                        bf16_t a = utils::cast<bf16_t>(static_cast<float>(alpha));
                        return op::scal(x, &a);
                  }
                  case DataType::F32: {
                        float a = static_cast<float>(alpha);
                        return op::scal(x, &a);
                  }
                  case DataType::F64: {
                        double a = alpha;
                        return op::scal(x, &a);
                  }
                  default:
                        throw std::runtime_error("scal only supports floating dtypes.");
                  }
            },
            py::arg("x"),
            py::arg("alpha"),
            R"doc(Element-wise scaling function.)doc");

      m.def(
            "scal_",
            [](Tensor y, Tensor x, double alpha) {
                  switch (x->dtype()) {
                  case DataType::F16: {
                        fp16_t a = utils::cast<fp16_t>(static_cast<float>(alpha));
                        op::scal_(y, x, &a);
                        return;
                  }
                  case DataType::BF16: {
                        bf16_t a = utils::cast<bf16_t>(static_cast<float>(alpha));
                        op::scal_(y, x, &a);
                        return;
                  }
                  case DataType::F32: {
                        float a = static_cast<float>(alpha);
                        op::scal_(y, x, &a);
                        return;
                  }
                  case DataType::F64: {
                        double a = alpha;
                        op::scal_(y, x, &a);
                        return;
                  }
                  default:
                        throw std::runtime_error("scal only supports floating dtypes.");
                  }
            },
            py::arg("y"),
            py::arg("x"),
            py::arg("alpha"),
            R"doc(In-place element-wise scaling function.)doc");
}

} // namespace infinicore::ops
