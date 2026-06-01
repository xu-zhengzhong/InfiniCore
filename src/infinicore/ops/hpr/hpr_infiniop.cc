#include "infinicore/ops/hpr.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::hpr_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Hpr, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, alpha, x, ap;
};

void *plan(const Tensor &alpha, const Tensor &x, Tensor ap, int uplo) {
    size_t seed = hash_combine(ap, alpha, x, uplo);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Hpr,
        seed,
        static_cast<infiniopBlasFillMode_t>(uplo),
        alpha->desc(), x->desc(), ap->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Hpr, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(alpha),
        graph::GraphTensor(x),
        graph::GraphTensor(ap)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopHpr(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->alpha->data(),
        planned->x->data(),
        planned->ap->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Hpr, &plan, &run, &cleanup);

} // namespace infinicore::op::hpr_impl::infiniop
