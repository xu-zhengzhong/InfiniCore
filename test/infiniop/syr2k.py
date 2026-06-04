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

_UPLO_TRANS_CASES = [(uplo, trans) for uplo in (0, 1) for trans in (0, 1)]

_DENSE_SHAPES = [
    # n, k
    (1, 1),
    (1, 7),
    (2, 1),
    (2, 2),
    (7, 3),
    (8, 8),
    (9, 17),
    (17, 9),
    (31, 32),
    (32, 31),
    (33, 64),
    (64, 33),
    (127, 128),
    (128, 127),
    (256, 256),
    (512, 512),
    (1024, 32),
    (32, 1024),
    (2048, 16),
    (16, 2048),
]

_STRIDED_SHAPES = [
    # n, k, a_stride_n, a_stride_t, b_stride_n, b_stride_t, c_stride
    (17, 9, (1, 24), (1, 16), (1, 28), (1, 18), None),
    (17, 9, (24, 1), (20, 1), (28, 1), (18, 1), (20, 1)),
    (31, 32, (1, 40), (1, 48), (1, 44), (1, 52), (1, 40)),
    (32, 31, (40, 1), (48, 1), (44, 1), (52, 1), (37, 1)),
    (128, 3, (1, 136), (1, 8), (1, 140), (1, 12), (1, 136)),
    (3, 128, (1, 8), (1, 136), (1, 12), (1, 140), (1, 8)),
]

_TEST_CASES = [
    # uplo, trans, n, k, a_stride, b_stride, c_stride
    (uplo, trans, n, k, None, None, None)
    for n, k in _DENSE_SHAPES
    for uplo, trans in _UPLO_TRANS_CASES
] + [
    (
        uplo,
        trans,
        n,
        k,
        a_stride_n if trans == 0 else a_stride_t,
        b_stride_n if trans == 0 else b_stride_t,
        c_stride,
    )
    for n, k, a_stride_n, a_stride_t, b_stride_n, b_stride_t, c_stride in _STRIDED_SHAPES
    for uplo, trans in _UPLO_TRANS_CASES
]

_TENSOR_DTYPES = [
    InfiniDtype.F32,
    # InfiniDtype.F64,
]

_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F64: {"atol": 1e-9, "rtol": 1e-9},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def _default_col_major_stride(rows):
    return (1, rows)


def _triangle_update(C, update, uplo):
    if uplo == 0:
        return torch.triu(update) + torch.tril(C, diagonal=-1)
    return torch.tril(update) + torch.triu(C, diagonal=1)


def test(
    handle,
    device,
    uplo,
    trans,
    n,
    k,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    dtype=torch.float32,
    sync=None,
):
    print(
        f"Testing Syr2k on {InfiniDeviceNames[device]} with uplo:{uplo} trans:{trans} n:{n} k:{k} "
        f"a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    ab_shape = (n, k) if trans == 0 else (k, n)
    if a_stride is None:
        a_stride = _default_col_major_stride(ab_shape[0])
    if b_stride is None:
        b_stride = _default_col_major_stride(ab_shape[0])
    if c_stride is None:
        c_stride = _default_col_major_stride(n)

    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    alpha = TestTensor(tuple(), None, dtype, device)
    beta = TestTensor(tuple(), None, dtype, device)
    A = TestTensor(ab_shape, a_stride, dtype, device, scale=0.5)
    B = TestTensor(ab_shape, b_stride, dtype, device, scale=0.5)
    C = TestTensor((n, n), c_stride, dtype, device)

    product = (
        torch.mm(A.torch_tensor(), B.torch_tensor().t())
        + torch.mm(B.torch_tensor(), A.torch_tensor().t())
        if trans == 0
        else torch.mm(A.torch_tensor().t(), B.torch_tensor())
        + torch.mm(B.torch_tensor().t(), A.torch_tensor())
    )
    update = alpha.torch_tensor() * product + beta.torch_tensor() * C.torch_tensor()
    C_ref = _triangle_update(C.torch_tensor(), update, uplo)
    C.update_torch_tensor(C_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSyr2kDescriptor(
            handle,
            ctypes.byref(descriptor),
            uplo,
            trans,
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
        LIBINFINIOP.infiniopGetSyr2kWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_syr2k():
        check_error(
            LIBINFINIOP.infiniopSyr2k(
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

    lib_syr2k()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(C.actual_tensor(), C.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(C.actual_tensor(), C.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: _triangle_update(
                C.torch_tensor(),
                alpha.torch_tensor()
                * (
                    torch.mm(A.torch_tensor(), B.torch_tensor().t())
                    + torch.mm(B.torch_tensor(), A.torch_tensor().t())
                    if trans == 0
                    else torch.mm(A.torch_tensor().t(), B.torch_tensor())
                    + torch.mm(B.torch_tensor().t(), A.torch_tensor())
                )
                + beta.torch_tensor() * C.torch_tensor(),
                uplo,
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_syr2k(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroySyr2kDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
