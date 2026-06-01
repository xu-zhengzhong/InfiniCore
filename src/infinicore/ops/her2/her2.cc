#include "infinicore/ops/her2.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Her2);

Her2::Her2(const Tensor &alpha, const Tensor &x, const Tensor &y, Tensor a, int uplo) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(alpha, x, y, a);
    INFINICORE_GRAPH_OP_DISPATCH(a->device().getType(), alpha, x, y, a, uplo);
}

void Her2::execute(const Tensor &alpha, const Tensor &x, const Tensor &y, Tensor a, int uplo) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Her2, alpha, x, y, a, uplo);
}

void her2_(const Tensor &alpha, const Tensor &x, const Tensor &y, Tensor a, int uplo) {
    Her2::execute(alpha, x, y, a, uplo);
}

} // namespace infinicore::op
