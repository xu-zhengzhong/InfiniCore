import ctypes
from ctypes import c_uint64

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    TestWorkspace,
    check_error,
    debug,
    get_args,
    get_test_devices,
    get_tolerance,
    infiniopOperatorDescriptor_t,
    profile_operation,
    test_operator,
)

# ==============================================================================
#  Configuration
# ==============================================================================

_TEST_CASES = [
    # uplo, n, x_stride
    (0, 1, None),
    (0, 5, None),
    (0, 17, (2,)),
    (0, 33, None),
    (0, 128, (3,)),
    (0, 1024, None),
    (1, 1, None),
    (1, 5, None),
    (1, 17, None),
    (1, 33, (2,)),
    (1, 128, (3,)),
    (1, 1024, None),
]

_TENSOR_DTYPES = [
    InfiniDtype.F32,
    # InfiniDtype.F64,
]

_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.F64: {"atol": 1e-9, "rtol": 1e-9},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def _packed_rank1_update(AP, x, alpha, uplo, n):
    update = alpha * torch.outer(x, x)
    if uplo == 0:
        rows, cols = torch.tril_indices(n, n, device=AP.device)
    else:
        rows, cols = torch.triu_indices(n, n, device=AP.device)

    return AP + update[cols, rows]


def test(
    handle,
    device,
    uplo,
    n,
    x_stride=None,
    dtype=torch.float32,
    sync=None,
):
    print(
        f"Testing Spr on {InfiniDeviceNames[device]} with uplo:{uplo} n:{n} "
        f"x_stride:{x_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    packed_len = n * (n + 1) // 2
    alpha = TestTensor(tuple(), None, dtype, device)
    x = TestTensor((n,), x_stride, dtype, device)
    AP = TestTensor((packed_len,), None, dtype, device)

    AP_ref = _packed_rank1_update(
        AP.torch_tensor(), x.torch_tensor(), alpha.torch_tensor(), uplo, n
    )
    AP.update_torch_tensor(AP_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSprDescriptor(
            handle,
            ctypes.byref(descriptor),
            uplo,
            alpha.descriptor,
            x.descriptor,
            AP.descriptor,
        )
    )

    for tensor in [alpha, x, AP]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetSprWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_spr():
        check_error(
            LIBINFINIOP.infiniopSpr(
                descriptor,
                workspace.data(),
                workspace_size.value,
                alpha.data(),
                x.data(),
                AP.data(),
                None,
            )
        )

    lib_spr()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(AP.actual_tensor(), AP.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(AP.actual_tensor(), AP.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: _packed_rank1_update(
                AP.torch_tensor(), x.torch_tensor(), alpha.torch_tensor(), uplo, n
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_spr(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroySprDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
