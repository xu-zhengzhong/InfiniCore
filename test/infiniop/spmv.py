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
    # uplo, n, x_stride, y_stride
    (0, 1, None, None),
    (0, 5, None, None),
    (0, 17, (2,), None),
    (0, 33, None, (2,)),
    (0, 128, (2,), (3,)),
    (0, 1024, None, None),
    (1, 1, None, None),
    (1, 5, None, None),
    (1, 17, None, (2,)),
    (1, 33, (2,), None),
    (1, 128, (3,), (2,)),
    (1, 1024, None, None),
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


def _packed_to_full(AP, uplo, n):
    matrix = torch.empty((n, n), dtype=AP.dtype, device=AP.device)
    if uplo == 0:
        rows, cols = torch.tril_indices(n, n, device=AP.device)
    else:
        rows, cols = torch.triu_indices(n, n, device=AP.device)
    matrix[cols, rows] = AP
    matrix[rows, cols] = AP
    return matrix


def test(
    handle,
    device,
    uplo,
    n,
    x_stride=None,
    y_stride=None,
    dtype=torch.float32,
    sync=None,
):
    print(
        f"Testing Spmv on {InfiniDeviceNames[device]} with uplo:{uplo} n:{n} "
        f"x_stride:{x_stride} y_stride:{y_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    packed_len = n * (n + 1) // 2
    alpha = TestTensor(tuple(), None, dtype, device)
    beta = TestTensor(tuple(), None, dtype, device)
    AP = TestTensor((packed_len,), None, dtype, device)
    x = TestTensor((n,), x_stride, dtype, device)
    y = TestTensor((n,), y_stride, dtype, device)

    matrix = _packed_to_full(AP.torch_tensor(), uplo, n)
    y_ref = alpha.torch_tensor() * torch.mv(matrix, x.torch_tensor())
    y_ref = y_ref + beta.torch_tensor() * y.torch_tensor()
    y.update_torch_tensor(y_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSpmvDescriptor(
            handle,
            ctypes.byref(descriptor),
            uplo,
            alpha.descriptor,
            AP.descriptor,
            x.descriptor,
            beta.descriptor,
            y.descriptor,
        )
    )

    for tensor in [alpha, beta, AP, x, y]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetSpmvWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_spmv():
        check_error(
            LIBINFINIOP.infiniopSpmv(
                descriptor,
                workspace.data(),
                workspace_size.value,
                alpha.data(),
                AP.data(),
                x.data(),
                beta.data(),
                y.data(),
                None,
            )
        )

    lib_spmv()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: torch.mv(matrix, x.torch_tensor())
            .mul_(alpha.torch_tensor())
            .add_(y.torch_tensor(), alpha=beta.torch_tensor().item()),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_spmv(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroySpmvDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
