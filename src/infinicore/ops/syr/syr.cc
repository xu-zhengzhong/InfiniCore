#include "infinicore/ops/syr.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Syr);

Syr::Syr(const Tensor &alpha, const Tensor &x, Tensor a, int uplo) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(alpha, x, a);
    INFINICORE_GRAPH_OP_DISPATCH(a->device().getType(), alpha, x, a, uplo);
}

void Syr::execute(const Tensor &alpha, const Tensor &x, Tensor a, int uplo) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Syr, alpha, x, a, uplo);
}

void syr_(const Tensor &alpha, const Tensor &x, Tensor a, int uplo) {
    Syr::execute(alpha, x, a, uplo);
}

} // namespace infinicore::op
