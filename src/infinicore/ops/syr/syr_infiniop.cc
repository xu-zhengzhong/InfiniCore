#include "infinicore/ops/syr.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::syr_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Syr, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, alpha, x, a;
};

void *plan(const Tensor &alpha, const Tensor &x, Tensor a, int uplo) {
    size_t seed = hash_combine(a, alpha, x, uplo);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Syr,
        seed,
        static_cast<infiniopBlasFillMode_t>(uplo),
        alpha->desc(), x->desc(), a->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Syr, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(alpha),
        graph::GraphTensor(x),
        graph::GraphTensor(a)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopSyr(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->alpha->data(),
        planned->x->data(),
        planned->a->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Syr, &plan, &run, &cleanup);

} // namespace infinicore::op::syr_impl::infiniop
