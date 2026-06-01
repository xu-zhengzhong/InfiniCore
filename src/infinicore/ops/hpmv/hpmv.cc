#include "infinicore/ops/hpmv.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Hpmv);

Hpmv::Hpmv(const Tensor &alpha, const Tensor &ap, const Tensor &x, const Tensor &beta, Tensor y, int uplo) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(alpha, ap, x, beta, y);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), alpha, ap, x, beta, y, uplo);
}

void Hpmv::execute(const Tensor &alpha, const Tensor &ap, const Tensor &x, const Tensor &beta, Tensor y, int uplo) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Hpmv, alpha, ap, x, beta, y, uplo);
}

void hpmv_(const Tensor &alpha, const Tensor &ap, const Tensor &x, const Tensor &beta, Tensor y, int uplo) {
    Hpmv::execute(alpha, ap, x, beta, y, uplo);
}

} // namespace infinicore::op
