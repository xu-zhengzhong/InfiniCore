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

_SIDE_UPLO_TRANS_DIAG_CASES = [
    (side, uplo, trans, diag)
    for side in (0, 1)
    for uplo in (0, 1)
    for trans in (0, 1)
    for diag in (0, 1)
]

_DENSE_SHAPES = [
    # m, n
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (3, 5),
    (5, 3),
    (8, 8),
    (17, 9),
    (31, 32),
    (32, 31),
    (65, 65),
    (127, 128),
    (128, 127),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (4096, 1),
    (1, 4096),
]

_STRIDED_SHAPES = [
    # m, n, a_stride_left, a_stride_right, b_stride
    (17, 9, (1, 24), (1, 16), None),
    (17, 9, (24, 1), (16, 1), (12, 1)),
    (31, 32, (1, 40), (1, 48), (1, 36)),
    (32, 31, (40, 1), (48, 1), (35, 1)),
    (4096, 2, (1, 4104), (1, 4), (1, 4100)),
    (2, 4096, (1, 4), (1, 4104), (1, 4)),
]

_TEST_CASES = [
    # side, uplo, trans, diag, m, n, a_stride, b_stride
    (side, uplo, trans, diag, m, n, None, None)
    for m, n in _DENSE_SHAPES
    for side, uplo, trans, diag in _SIDE_UPLO_TRANS_DIAG_CASES
] + [
    (
        side,
        uplo,
        trans,
        diag,
        m,
        n,
        a_stride_left if side == 0 else a_stride_right,
        b_stride,
    )
    for m, n, a_stride_left, a_stride_right, b_stride in _STRIDED_SHAPES
    for side, uplo, trans, diag in _SIDE_UPLO_TRANS_DIAG_CASES
]

_TENSOR_DTYPES = [
    InfiniDtype.F32,
    # InfiniDtype.F64,
]

_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-4, "rtol": 1e-4},
    InfiniDtype.F64: {"atol": 1e-9, "rtol": 1e-9},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def _full_from_triangle(A, uplo, diag):
    if uplo == 0:
        triangular = torch.triu(A)
    else:
        triangular = torch.tril(A)
    if diag == 1:
        triangular = triangular.clone()
        triangular.diagonal().fill_(1)
    return triangular


def _default_col_major_stride(rows):
    return (1, rows)


def test(
    handle,
    device,
    side,
    uplo,
    trans,
    diag,
    m,
    n,
    a_stride=None,
    b_stride=None,
    dtype=torch.float32,
    sync=None,
):
    print(
        f"Testing Trmm on {InfiniDeviceNames[device]} with side:{side} uplo:{uplo} trans:{trans} "
        f"diag:{diag} m:{m} n:{n} a_stride:{a_stride} b_stride:{b_stride} "
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    dim_a = m if side == 0 else n
    if a_stride is None:
        a_stride = _default_col_major_stride(dim_a)
    if b_stride is None:
        b_stride = _default_col_major_stride(m)

    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    alpha = TestTensor(tuple(), None, dtype, device)
    A = TestTensor((dim_a, dim_a), a_stride, dtype, device, scale=0.5)
    B = TestTensor((m, n), b_stride, dtype, device, scale=0.5)

    matrix = _full_from_triangle(A.torch_tensor(), uplo, diag)
    op_a = matrix if trans == 0 else matrix.t()
    b_input = B.torch_tensor().clone()
    b_ref = alpha.torch_tensor() * (
        torch.mm(op_a, b_input) if side == 0 else torch.mm(b_input, op_a)
    )
    B.update_torch_tensor(b_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateTrmmDescriptor(
            handle,
            ctypes.byref(descriptor),
            side,
            uplo,
            trans,
            diag,
            alpha.descriptor,
            A.descriptor,
            B.descriptor,
        )
    )

    for tensor in [alpha, A, B]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetTrmmWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_trmm():
        check_error(
            LIBINFINIOP.infiniopTrmm(
                descriptor,
                workspace.data(),
                workspace_size.value,
                alpha.data(),
                A.data(),
                B.data(),
                None,
            )
        )

    lib_trmm()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(B.actual_tensor(), B.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(B.actual_tensor(), B.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: alpha.torch_tensor()
            * (torch.mm(op_a, b_input) if side == 0 else torch.mm(b_input, op_a)),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_trmm(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyTrmmDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
