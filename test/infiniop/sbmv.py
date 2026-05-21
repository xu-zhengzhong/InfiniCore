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
    # uplo, n, k, a_stride, x_stride, y_stride
    (0, 1, 0, None, None, None),
    (0, 5, 0, None, None, None),
    (0, 5, 1, None, None, None),
    (0, 17, 3, None, (2,), None),
    (0, 33, 4, (1, 8), None, (2,)),
    (0, 33, 32, None, (2,), None),
    (0, 128, 7, None, (2,), (3,)),
    (0, 1024, 2, None, None, None),
    (1, 1, 0, None, None, None),
    (1, 5, 0, None, None, None),
    (1, 5, 1, None, None, None),
    (1, 17, 3, None, None, (2,)),
    (1, 33, 4, (1, 8), (2,), None),
    (1, 33, 32, None, None, None),
    (1, 128, 7, None, (3,), (2,)),
    (1, 1024, 2, None, None, None),
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


def _full_from_symmetric_band(A, n, k, uplo):
    full = torch.zeros((n, n), dtype=A.dtype, device=A.device)
    if uplo == 0:
        for j in range(n):
            i_begin = max(0, j - k)
            for i in range(i_begin, j + 1):
                value = A[k + i - j, j]
                full[i, j] = value
                full[j, i] = value
    else:
        for j in range(n):
            i_end = min(n, j + k + 1)
            for i in range(j, i_end):
                value = A[i - j, j]
                full[i, j] = value
                full[j, i] = value
    return full


def test(
    handle,
    device,
    uplo,
    n,
    k,
    a_stride=None,
    x_stride=None,
    y_stride=None,
    dtype=torch.float32,
    sync=None,
):
    print(
        f"Testing Sbmv on {InfiniDeviceNames[device]} with uplo:{uplo} n:{n} k:{k} "
        f"a_stride:{a_stride} x_stride:{x_stride} y_stride:{y_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    if a_stride is None:
        a_stride = (1, k + 1)

    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    alpha = TestTensor(tuple(), None, dtype, device)
    beta = TestTensor(tuple(), None, dtype, device)
    A = TestTensor((k + 1, n), a_stride, dtype, device)
    x = TestTensor((n,), x_stride, dtype, device)
    y = TestTensor((n,), y_stride, dtype, device)

    matrix = _full_from_symmetric_band(A.torch_tensor(), n, k, uplo)
    y_ref = alpha.torch_tensor() * torch.mv(matrix, x.torch_tensor())
    y_ref = y_ref + beta.torch_tensor() * y.torch_tensor()
    y.update_torch_tensor(y_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSbmvDescriptor(
            handle,
            ctypes.byref(descriptor),
            uplo,
            k,
            alpha.descriptor,
            A.descriptor,
            x.descriptor,
            beta.descriptor,
            y.descriptor,
        )
    )

    for tensor in [alpha, beta, A, x, y]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetSbmvWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_sbmv():
        check_error(
            LIBINFINIOP.infiniopSbmv(
                descriptor,
                workspace.data(),
                workspace_size.value,
                alpha.data(),
                A.data(),
                x.data(),
                beta.data(),
                y.data(),
                None,
            )
        )

    lib_sbmv()

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
            "    lib", lambda: lib_sbmv(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroySbmvDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
