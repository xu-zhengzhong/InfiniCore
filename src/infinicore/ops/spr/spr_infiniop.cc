#include "infinicore/ops/spr.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::spr_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Spr, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, alpha, x, ap;
};

void *plan(const Tensor &alpha, const Tensor &x, Tensor ap, int uplo) {
    size_t seed = hash_combine(ap, alpha, x, uplo);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Spr,
        seed,
        static_cast<infiniopBlasFillMode_t>(uplo),
        alpha->desc(), x->desc(), ap->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Spr, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(alpha),
        graph::GraphTensor(x),
        graph::GraphTensor(ap)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopSpr(
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

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Spr, &plan, &run, &cleanup);

} // namespace infinicore::op::spr_impl::infiniop
