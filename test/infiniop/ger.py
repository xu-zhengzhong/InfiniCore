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
    # m, n, a_stride, x_stride, y_stride
    (1, 1, None, None, None),
    (3, 4, (1, 3), None, None),
    (4, 5, (8, 1), (2,), (3,)),
    (7, 3, (1, 9), (2,), (3,)),
    (32, 17, None, None, None),
    (64, 33, (1, 66), (2,), (2,)),
    (16, 5632, None, (2,), (2,)),
    (5632, 33, (1, 5632), None, None),
    (2048, 2560, (1, 4096), None, None),
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


def test(
    handle,
    device,
    m,
    n,
    a_stride=None,
    x_stride=None,
    y_stride=None,
    dtype=torch.float32,
    sync=None,
):
    print(
        f"Testing Ger on {InfiniDeviceNames[device]} with m:{m} n:{n} "
        f"a_stride:{a_stride} x_stride:{x_stride} y_stride:{y_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    alpha = TestTensor(tuple(), None, dtype, device)
    x = TestTensor((m,), x_stride, dtype, device)
    y = TestTensor((n,), y_stride, dtype, device)
    A = TestTensor((m, n), a_stride, dtype, device)

    A_ref = A.torch_tensor() + alpha.torch_tensor() * torch.outer(
        x.torch_tensor(), y.torch_tensor()
    )
    A.update_torch_tensor(A_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateGerDescriptor(
            handle,
            ctypes.byref(descriptor),
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
        LIBINFINIOP.infiniopGetGerWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_ger():
        check_error(
            LIBINFINIOP.infiniopGer(
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

    lib_ger()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(A.actual_tensor(), A.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(A.actual_tensor(), A.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: A.torch_tensor().add_(
                torch.outer(x.torch_tensor(), y.torch_tensor()),
                alpha=alpha.torch_tensor().item(),
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_ger(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyGerDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
