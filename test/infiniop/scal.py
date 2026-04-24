import torch
import ctypes
from ctypes import c_uint64
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

# ==============================================================================
#  Configuration
# ==============================================================================

_TEST_CASES = [
    # n, x_stride
    (3, None),
    (8, (2,)),
    (32, None),
    (257, (3,)),
    (65535, None),
]

_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16, InfiniDtype.F32]

_TOLERANCE_MAP = {
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.F64: {"atol": 1e-9, "rtol": 1e-9},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def scal(x, alpha):
    x.mul_(alpha)


def test(
    handle,
    device,
    n,
    x_stride=None,
    dtype=torch.float16,
    sync=None,
):
    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    alpha = TestTensor(tuple(), None, dtype, device)
    x = TestTensor((n,), x_stride, dtype, device)

    print(
        f"Testing scal on {InfiniDeviceNames[device]} with n:{n} x_stride:{x_stride} "
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    scal(x.torch_tensor(), alpha.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateScalDescriptor(
            handle,
            ctypes.byref(descriptor),
            alpha.descriptor,
            x.descriptor,
        )
    )

    for tensor in [alpha, x]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetScalWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_scal():
        check_error(
            LIBINFINIOP.infiniopScal(
                descriptor,
                workspace.data(),
                workspace.size(),
                alpha.data(),
                x.data(),
                None,
            )
        )

    lib_scal()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(x.actual_tensor(), x.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(x.actual_tensor(), x.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: scal(x.torch_tensor(), alpha.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_scal(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyScalDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92m  Test passed!  \033[0m")