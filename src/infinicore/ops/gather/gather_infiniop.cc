#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/gather.hpp"
#include <infiniop.h>

namespace infinicore::op::gather_impl::infiniop {

thread_local common::OpCache<size_t, infiniopGatherDescriptor_t> caches(
    100,
    [](infiniopGatherDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyGatherDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, Tensor index, int dim) {
    // dim also participates in cache key
    size_t seed = hash_combine(output, input, index, static_cast<size_t>(dim));

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopGatherDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateGatherDescriptor(
            context::getInfiniopHandle(output->device()),
            &desc,
            output->desc(),
            input->desc(),
            index->desc(),
            dim));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetGatherWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopGather(
        desc,
        workspace->data(), workspace_size,
        output->data(),
        input->data(),
        index->data(),
        context::getStream()));
}

static bool registered = []() {
    Gather::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::gather_impl::infiniop