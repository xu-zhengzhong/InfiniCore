#include "infinicore/ops/syr2.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Syr2);

Syr2::Syr2(const Tensor &alpha, const Tensor &x, const Tensor &y, Tensor a, int uplo) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(alpha, x, y, a);
    INFINICORE_GRAPH_OP_DISPATCH(a->device().getType(), alpha, x, y, a, uplo);
}

void Syr2::execute(const Tensor &alpha, const Tensor &x, const Tensor &y, Tensor a, int uplo) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Syr2, alpha, x, y, a, uplo);
}

void syr2_(const Tensor &alpha, const Tensor &x, const Tensor &y, Tensor a, int uplo) {
    Syr2::execute(alpha, x, y, a, uplo);
}

} // namespace infinicore::op
