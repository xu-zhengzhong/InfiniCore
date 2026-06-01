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
    # uplo, trans, diag, n, x_stride
    (0, 0, 0, 1, None),
    (0, 0, 0, 5, None),
    (0, 0, 1, 5, None),
    (0, 1, 0, 5, None),
    (0, 1, 1, 17, (2,)),
    (0, 0, 0, 128, (3,)),
    (0, 1, 0, 1024, None),
    (1, 0, 0, 1, None),
    (1, 0, 0, 5, None),
    (1, 0, 1, 5, None),
    (1, 1, 0, 5, None),
    (1, 1, 1, 17, (2,)),
    (1, 0, 0, 128, (3,)),
    (1, 1, 0, 1024, None),
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


def _packed_to_triangular(AP, uplo, diag, n):
    matrix = torch.zeros((n, n), dtype=AP.dtype, device=AP.device)
    if uplo == 0:
        rows, cols = torch.tril_indices(n, n, device=AP.device)
    else:
        rows, cols = torch.triu_indices(n, n, device=AP.device)
    matrix[cols, rows] = AP
    if diag == 1:
        matrix.diagonal().fill_(1)
    return matrix


def test(
    handle,
    device,
    uplo,
    trans,
    diag,
    n,
    x_stride=None,
    dtype=torch.float32,
    sync=None,
):
    print(
        f"Testing Tpmv on {InfiniDeviceNames[device]} with uplo:{uplo} trans:{trans} diag:{diag} n:{n} "
        f"x_stride:{x_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    packed_len = n * (n + 1) // 2
    AP = TestTensor((packed_len,), None, dtype, device)
    x = TestTensor((n,), x_stride, dtype, device)
    x_input = x.torch_tensor().clone()

    matrix = _packed_to_triangular(AP.torch_tensor(), uplo, diag, n)
    op_matrix = matrix if trans == 0 else matrix.t()
    x_ref = torch.mv(op_matrix, x_input)
    x.update_torch_tensor(x_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateTpmvDescriptor(
            handle,
            ctypes.byref(descriptor),
            uplo,
            trans,
            diag,
            AP.descriptor,
            x.descriptor,
        )
    )

    for tensor in [AP, x]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetTpmvWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_tpmv():
        check_error(
            LIBINFINIOP.infiniopTpmv(
                descriptor,
                workspace.data(),
                workspace_size.value,
                AP.data(),
                x.data(),
                None,
            )
        )

    lib_tpmv()

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
            "    lib", lambda: lib_tpmv(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyTpmvDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
