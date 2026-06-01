#include "infinicore/ops/tbsv.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Tbsv);

Tbsv::Tbsv(const Tensor &a, Tensor x, int uplo, int trans, int diag, size_t k) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(a, x);
    INFINICORE_GRAPH_OP_DISPATCH(x->device().getType(), a, x, uplo, trans, diag, k);
}

void Tbsv::execute(const Tensor &a, Tensor x, int uplo, int trans, int diag, size_t k) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Tbsv, a, x, uplo, trans, diag, k);
}

void tbsv_(const Tensor &a, Tensor x, int uplo, int trans, int diag, size_t k) {
    Tbsv::execute(a, x, uplo, trans, diag, k);
}

} // namespace infinicore::op
