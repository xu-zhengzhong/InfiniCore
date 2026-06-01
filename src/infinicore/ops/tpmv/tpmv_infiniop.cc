#include "infinicore/ops/tpmv.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::tpmv_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Tpmv, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, ap, x;
};

void *plan(const Tensor &ap, Tensor x, int uplo, int trans, int diag) {
    size_t seed = hash_combine(x, ap, uplo, trans, diag);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Tpmv,
        seed,
        static_cast<infiniopBlasFillMode_t>(uplo),
        static_cast<infiniopBlasOperation_t>(trans),
        static_cast<infiniopBlasDiagType_t>(diag),
        ap->desc(), x->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Tpmv, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(ap),
        graph::GraphTensor(x)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopTpmv(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->ap->data(),
        planned->x->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Tpmv, &plan, &run, &cleanup);

} // namespace infinicore::op::tpmv_impl::infiniop
