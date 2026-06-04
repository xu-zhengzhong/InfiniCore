#include "infinicore/ops/trsm.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::trsm_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Trsm, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, alpha, a, b;
};

void *plan(const Tensor &a, const Tensor &alpha, Tensor b, int side, int uplo, int trans, int diag) {
    size_t seed = hash_combine(b, a, alpha, side, uplo, trans, diag);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Trsm,
        seed,
        static_cast<infiniopBlasSideMode_t>(side),
        static_cast<infiniopBlasFillMode_t>(uplo),
        static_cast<infiniopBlasOperation_t>(trans),
        static_cast<infiniopBlasDiagType_t>(diag),
        alpha->desc(), a->desc(), b->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Trsm, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(alpha),
        graph::GraphTensor(a),
        graph::GraphTensor(b)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopTrsm(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->alpha->data(),
        planned->a->data(),
        planned->b->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Trsm, &plan, &run, &cleanup);

} // namespace infinicore::op::trsm_impl::infiniop
