#include "infinicore/ops/hemv.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Hemv);

Hemv::Hemv(const Tensor &alpha, const Tensor &a, const Tensor &x, const Tensor &beta, Tensor y, int uplo) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(alpha, a, x, beta, y);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), alpha, a, x, beta, y, uplo);
}

void Hemv::execute(const Tensor &alpha, const Tensor &a, const Tensor &x, const Tensor &beta, Tensor y, int uplo) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Hemv, alpha, a, x, beta, y, uplo);
}

void hemv_(const Tensor &alpha, const Tensor &a, const Tensor &x, const Tensor &beta, Tensor y, int uplo) {
    Hemv::execute(alpha, a, x, beta, y, uplo);
}

} // namespace infinicore::op
