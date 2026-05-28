#ifndef __SPVV_METAX_H__
#define __SPVV_METAX_H__

#include "../spvv.h"

SPVV_DESCRIPTOR(metax);

namespace op::spvv::metax {
infiniStatus_t launchApplyAlphaBeta(void *y, const void *dot, float alpha, float beta, void *stream);
} // namespace op::spvv::metax

#endif // __SPVV_METAX_H__
