#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/narrow.hpp"
#include <infiniop.h>

namespace infinicore::op:: narrow_impl:: infiniop {

thread_local common::OpCache<size_t, infiniopNarrowDescriptor_t> caches(
    100,
    [](infiniopNarrowDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyNarrowDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, int dim, int64_t start, int64_t length) {
    size_t seed = hash_combine(output, input, dim, start, length);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches. getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopNarrowDescriptor_t desc = nullptr;

    if (! desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateNarrowDescriptor(
            context::getInfiniopHandle(output->device()), &desc,
            output->desc(), input->desc(), dim, start, length));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetNarrowWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopNarrow(
        desc, 
        workspace->data(), workspace_size,
        output->data(), input->data(), 
        context::getStream()));
}

static bool registered = []() {
    Narrow::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::narrow_impl::infiniop