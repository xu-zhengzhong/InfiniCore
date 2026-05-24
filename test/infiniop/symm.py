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

_SIDE_UPLO_CASES = [(side, uplo) for side in (0, 1) for uplo in (0, 1)]

_DENSE_SHAPES = [
    # m, n
    (1, 1),
    (1, 2),
    (1, 7),
    (2, 1),
    (7, 1),
    (2, 2),
    (3, 5),
    (5, 3),
    (8, 8),
    (9, 17),
    (17, 9),
    (31, 32),
    (32, 31),
    (32, 32),
    (33, 64),
    (64, 33),
    (65, 65),
    (127, 128),
    (128, 127),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (4096, 1),
    (1, 4096),
    (4096, 3),
    (3, 4096),
    (4097, 1),
    (1, 4097),
]

_STRIDED_SHAPES = [
    # m, n, a_stride_left, a_stride_right, b_stride, c_stride
    (17, 9, (1, 24), (1, 16), None, None),
    (17, 9, (24, 1), (16, 1), (12, 1), (13, 1)),
    (31, 32, (1, 40), (1, 48), (1, 36), (1, 40)),
    (32, 31, (40, 1), (48, 1), (35, 1), (37, 1)),
    (4096, 2, (1, 4104), (1, 4), (1, 4100), (1, 4104)),
    (2, 4096, (1, 4), (1, 4104), (1, 4), (1, 4)),
]

_TEST_CASES = [
    # side, uplo, m, n, a_stride, b_stride, c_stride
    (side, uplo, m, n, None, None, None)
    for m, n in _DENSE_SHAPES
    for side, uplo in _SIDE_UPLO_CASES
] + [
    (
        side,
        uplo,
        m,
        n,
        a_stride_left if side == 0 else a_stride_right,
        b_stride,
        c_stride,
    )
    for m, n, a_stride_left, a_stride_right, b_stride, c_stride in _STRIDED_SHAPES
    for side, uplo in _SIDE_UPLO_CASES
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


def _full_from_triangle(A, uplo):
    if uplo == 0:
        return torch.triu(A) + torch.triu(A, diagonal=1).t()
    return torch.tril(A) + torch.tril(A, diagonal=-1).t()


def _default_col_major_stride(rows):
    return (1, rows)


def test(
    handle,
    device,
    side,
    uplo,
    m,
    n,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    dtype=torch.float32,
    sync=None,
):
    print(
        f"Testing Symm on {InfiniDeviceNames[device]} with side:{side} uplo:{uplo} m:{m} n:{n} "
        f"a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    dim_a = m if side == 0 else n
    if a_stride is None:
        a_stride = _default_col_major_stride(dim_a)
    if b_stride is None:
        b_stride = _default_col_major_stride(m)
    if c_stride is None:
        c_stride = _default_col_major_stride(m)

    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    alpha = TestTensor(tuple(), None, dtype, device)
    beta = TestTensor(tuple(), None, dtype, device)
    A = TestTensor((dim_a, dim_a), a_stride, dtype, device)
    B = TestTensor((m, n), b_stride, dtype, device, scale=0.5)
    C = TestTensor((m, n), c_stride, dtype, device)

    matrix = _full_from_triangle(A.torch_tensor(), uplo)
    product = (
        torch.mm(matrix, B.torch_tensor())
        if side == 0
        else torch.mm(B.torch_tensor(), matrix)
    )
    c_ref = alpha.torch_tensor() * product + beta.torch_tensor() * C.torch_tensor()
    C.update_torch_tensor(c_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSymmDescriptor(
            handle,
            ctypes.byref(descriptor),
            side,
            uplo,
            alpha.descriptor,
            A.descriptor,
            B.descriptor,
            beta.descriptor,
            C.descriptor,
        )
    )

    for tensor in [alpha, beta, A, B, C]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetSymmWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_symm():
        check_error(
            LIBINFINIOP.infiniopSymm(
                descriptor,
                workspace.data(),
                workspace_size.value,
                alpha.data(),
                A.data(),
                B.data(),
                beta.data(),
                C.data(),
                None,
            )
        )

    lib_symm()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(C.actual_tensor(), C.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(C.actual_tensor(), C.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: alpha.torch_tensor()
            * (
                torch.mm(matrix, B.torch_tensor())
                if side == 0
                else torch.mm(B.torch_tensor(), matrix)
            )
            + beta.torch_tensor() * C.torch_tensor(),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_symm(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroySymmDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
