#pragma once

#include <pybind11/pybind11.h>

#include "../../../utils/custom_types.h"
#include "infinicore/ops/scal.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_scal(py::module &m) {
    m.def(
        "scal_",
        [](Tensor x, double alpha) {
            switch (x->dtype()) {
            case DataType::F16: {
                fp16_t a = utils::cast<fp16_t>(static_cast<float>(alpha));
                op::scal_(x, &a);
                return;
            }
            case DataType::BF16: {
                bf16_t a = utils::cast<bf16_t>(static_cast<float>(alpha));
                op::scal_(x, &a);
                return;
            }
            case DataType::F32: {
                float a = static_cast<float>(alpha);
                op::scal_(x, &a);
                return;
            }
            case DataType::F64: {
                double a = alpha;
                op::scal_(x, &a);
                return;
            }
            default:
                throw std::runtime_error("scal only supports floating dtypes.");
            }
        },
        py::arg("x"),
        py::arg("alpha"),
        R"doc(BLAS Level1 in-place scal function.)doc");
}

} // namespace infinicore::ops
