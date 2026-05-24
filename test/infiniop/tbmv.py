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
    # uplo, trans, diag, n, k, a_stride, x_stride
    (0, 0, 0, 1, 0, None, None),
    (0, 0, 0, 5, 0, None, None),
    (0, 0, 1, 5, 0, None, None),
    (0, 0, 0, 5, 1, None, None),
    (0, 0, 1, 5, 3, None, None),
    (0, 1, 0, 5, 2, None, None),
    (0, 1, 1, 17, 4, None, (2,)),
    (0, 0, 0, 33, 7, (1, 12), None),
    (0, 1, 1, 33, 8, (1, 16), (2,)),
    (0, 0, 0, 128, 5, None, (3,)),
    (0, 1, 0, 1024, 31, None, None),
    (1, 0, 0, 1, 0, None, None),
    (1, 0, 0, 5, 0, None, None),
    (1, 0, 1, 5, 0, None, None),
    (1, 0, 0, 5, 1, None, None),
    (1, 0, 1, 5, 3, None, None),
    (1, 1, 0, 5, 2, None, None),
    (1, 1, 1, 17, 4, None, (2,)),
    (1, 0, 0, 33, 7, (1, 12), (2,)),
    (1, 1, 1, 33, 8, (1, 16), None),
    (1, 0, 0, 128, 5, None, (3,)),
    (1, 1, 0, 1024, 31, None, None),
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


def _full_from_band(A, uplo, diag, n, k):
    full = torch.zeros((n, n), dtype=A.dtype, device=A.device)
    if uplo == 0:
        for j in range(n):
            i_begin = max(0, j - k)
            for i in range(i_begin, j + 1):
                full[i, j] = 1 if diag == 1 and i == j else A[k + i - j, j]
    else:
        for j in range(n):
            i_end = min(n, j + k + 1)
            for i in range(j, i_end):
                full[i, j] = 1 if diag == 1 and i == j else A[i - j, j]
    return full


def test(
    handle,
    device,
    uplo,
    trans,
    diag,
    n,
    k,
    a_stride=None,
    x_stride=None,
    dtype=torch.float32,
    sync=None,
):
    print(
        f"Testing Tbmv on {InfiniDeviceNames[device]} with uplo:{uplo} trans:{trans} diag:{diag} n:{n} k:{k} "
        f"a_stride:{a_stride} x_stride:{x_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    if a_stride is None:
        a_stride = (1, k + 1)

    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    A = TestTensor((k + 1, n), a_stride, dtype, device)
    x = TestTensor((n,), x_stride, dtype, device)
    x_input = x.torch_tensor().clone()

    full = _full_from_band(A.torch_tensor(), uplo, diag, n, k)
    op_matrix = full if trans == 0 else full.t()
    x_ref = torch.mv(op_matrix, x_input)
    x.update_torch_tensor(x_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateTbmvDescriptor(
            handle,
            ctypes.byref(descriptor),
            uplo,
            trans,
            diag,
            k,
            A.descriptor,
            x.descriptor,
        )
    )

    for tensor in [A, x]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetTbmvWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_tbmv():
        check_error(
            LIBINFINIOP.infiniopTbmv(
                descriptor,
                workspace.data(),
                workspace_size.value,
                A.data(),
                x.data(),
                None,
            )
        )

    lib_tbmv()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(x.actual_tensor(), x.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(x.actual_tensor(), x.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: torch.mv(op_matrix, x_input),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_tbmv(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyTbmvDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
