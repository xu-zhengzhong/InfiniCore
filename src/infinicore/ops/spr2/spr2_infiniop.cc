#include "infinicore/ops/spr2.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::spr2_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Spr2, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, alpha, x, y, ap;
};

void *plan(const Tensor &alpha, const Tensor &x, const Tensor &y, Tensor ap, int uplo) {
    size_t seed = hash_combine(ap, alpha, x, y, uplo);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Spr2,
        seed,
        static_cast<infiniopBlasFillMode_t>(uplo),
        alpha->desc(), x->desc(), y->desc(), ap->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Spr2, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(alpha),
        graph::GraphTensor(x),
        graph::GraphTensor(y),
        graph::GraphTensor(ap)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopSpr2(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->alpha->data(),
        planned->x->data(),
        planned->y->data(),
        planned->ap->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Spr2, &plan, &run, &cleanup);

} // namespace infinicore::op::spr2_impl::infiniop
