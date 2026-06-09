#include "infinicore/ops/blas_spmv.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(BlasSpmv);

BlasSpmv::BlasSpmv(const Tensor &alpha, const Tensor &ap, const Tensor &x, const Tensor &beta, Tensor y, int uplo) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(alpha, ap, x, beta, y);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), alpha, ap, x, beta, y, uplo);
}

void BlasSpmv::execute(const Tensor &alpha, const Tensor &ap, const Tensor &x, const Tensor &beta, Tensor y, int uplo) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(BlasSpmv, alpha, ap, x, beta, y, uplo);
}

void blas_spmv_(const Tensor &alpha, const Tensor &ap, const Tensor &x, const Tensor &beta, Tensor y, int uplo) {
    BlasSpmv::execute(alpha, ap, x, beta, y, uplo);
}

} // namespace infinicore::op
