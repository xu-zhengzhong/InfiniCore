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
    # uplo, n, a_stride, x_stride, y_stride
    (0, 1, None, None, None),
    (0, 5, None, None, None),
    (0, 5, (5, 1), None, (2,)),
    (0, 17, None, (2,), None),
    (0, 33, (1, 40), None, (2,)),
    (0, 33, (40, 1), (2,), None),
    (0, 128, None, (2,), (3,)),
    (0, 1024, None, None, None),
    (1, 1, None, None, None),
    (1, 5, None, None, None),
    (1, 5, (5, 1), (2,), None),
    (1, 17, None, None, (2,)),
    (1, 33, (1, 40), (2,), None),
    (1, 33, (40, 1), None, (2,)),
    (1, 128, None, (3,), (2,)),
    (1, 1024, None, None, None),
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


def _triangle_update(A, update, uplo):
    if uplo == 0:
        return torch.triu(A + update) + torch.tril(A, diagonal=-1)
    return torch.tril(A + update) + torch.triu(A, diagonal=1)


def test(
    handle,
    device,
    uplo,
    n,
    a_stride=None,
    x_stride=None,
    y_stride=None,
    dtype=torch.float32,
    sync=None,
):
    print(
        f"Testing Syr2 on {InfiniDeviceNames[device]} with uplo:{uplo} n:{n} "
        f"a_stride:{a_stride} x_stride:{x_stride} y_stride:{y_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    if a_stride is None:
        a_stride = (1, n)

    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    alpha = TestTensor(tuple(), None, dtype, device)
    x = TestTensor((n,), x_stride, dtype, device)
    y = TestTensor((n,), y_stride, dtype, device)
    A = TestTensor((n, n), a_stride, dtype, device)

    update = alpha.torch_tensor() * (
        torch.outer(x.torch_tensor(), y.torch_tensor())
        + torch.outer(y.torch_tensor(), x.torch_tensor())
    )
    A_ref = _triangle_update(A.torch_tensor(), update, uplo)
    A.update_torch_tensor(A_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSyr2Descriptor(
            handle,
            ctypes.byref(descriptor),
            uplo,
            alpha.descriptor,
            x.descriptor,
            y.descriptor,
            A.descriptor,
        )
    )

    for tensor in [alpha, x, y, A]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetSyr2WorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_syr2():
        check_error(
            LIBINFINIOP.infiniopSyr2(
                descriptor,
                workspace.data(),
                workspace_size.value,
                alpha.data(),
                x.data(),
                y.data(),
                A.data(),
                None,
            )
        )

    lib_syr2()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(A.actual_tensor(), A.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(A.actual_tensor(), A.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: _triangle_update(
                A.torch_tensor(),
                alpha.torch_tensor()
                * (
                    torch.outer(x.torch_tensor(), y.torch_tensor())
                    + torch.outer(y.torch_tensor(), x.torch_tensor())
                ),
                uplo,
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_syr2(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroySyr2Descriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
