#include "infinicore/ops/hpr.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Hpr);

Hpr::Hpr(const Tensor &alpha, const Tensor &x, Tensor ap, int uplo) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(alpha, x, ap);
    INFINICORE_GRAPH_OP_DISPATCH(ap->device().getType(), alpha, x, ap, uplo);
}

void Hpr::execute(const Tensor &alpha, const Tensor &x, Tensor ap, int uplo) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Hpr, alpha, x, ap, uplo);
}

void hpr_(const Tensor &alpha, const Tensor &x, Tensor ap, int uplo) {
    Hpr::execute(alpha, x, ap, uplo);
}

} // namespace infinicore::op
