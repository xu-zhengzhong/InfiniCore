#include "infinicore/ops/her.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Her);

Her::Her(const Tensor &alpha, const Tensor &x, Tensor a, int uplo) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(alpha, x, a);
    INFINICORE_GRAPH_OP_DISPATCH(a->device().getType(), alpha, x, a, uplo);
}

void Her::execute(const Tensor &alpha, const Tensor &x, Tensor a, int uplo) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Her, alpha, x, a, uplo);
}

void her_(const Tensor &alpha, const Tensor &x, Tensor a, int uplo) {
    Her::execute(alpha, x, a, uplo);
}

} // namespace infinicore::op
