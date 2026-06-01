#include "infinicore/ops/tbsv.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::tbsv_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Tbsv, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, a, x;
};

void *plan(const Tensor &a, Tensor x, int uplo, int trans, int diag, size_t k) {
    size_t seed = hash_combine(x, a, uplo, trans, diag, k);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Tbsv,
        seed,
        static_cast<infiniopBlasFillMode_t>(uplo),
        static_cast<infiniopBlasOperation_t>(trans),
        static_cast<infiniopBlasDiagType_t>(diag),
        k,
        a->desc(), x->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Tbsv, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(a),
        graph::GraphTensor(x)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopTbsv(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->a->data(),
        planned->x->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Tbsv, &plan, &run, &cleanup);

} // namespace infinicore::op::tbsv_impl::infiniop
