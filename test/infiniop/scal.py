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
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = [
    # shape, x_stride, y_stride, alpha
    ((13, 4), None, None, 2.5),
    ((13, 4), (10, 1), (10, 1), 1.3),
    ((13, 4), (16, 4), None, 1.0),
    ((13, 4, 4), None, None, 8.2),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), 0.5),
    ((16, 5632), None, None, 2.0),
    ((16, 5632), None, (13312, 1), 1.5),
    ((13, 16, 2), (128, 4, 1), (64, 4, 1), 0.8),
]

# Data types used for testing
_TENSOR_DTYPES = [
    InfiniDtype.F16,
    InfiniDtype.F32,
    InfiniDtype.BF16,
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def scal(y, x, alpha):
    torch.mul(x, alpha, out=y)


def test(
    handle,
    device,
    shape,
    y_stride=None,
    x_stride=None,
    alpha=2.5,
    dtype=torch.float16,
    sync=None,
):
    y = TestTensor(shape, y_stride, dtype, device, mode="ones")
    x = TestTensor(shape, x_stride, dtype, device)

    if y.is_broadcast():
        return

    print(
        f"Testing Scal on {InfiniDeviceNames[device]} with shape:{shape} y_stride:{y_stride} x_stride:{x_stride} alpha:{alpha} "
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    # Compute PyTorch reference
    scal(y.torch_tensor(), x.torch_tensor(), alpha)

    if sync is not None:
        sync()

    # Create Descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateScalDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            x.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    y.destroy_desc()
    x.destroy_desc()

    # Allocate Workspace
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetScalWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, y.device)

    # Execute C library op
    def lib_scal():
        check_error(
            LIBINFINIOP.infiniopScal(
                descriptor,
                workspace.data(),
                workspace.size(),
                y.data(),
                x.data(),
                alpha,
                None,
            )
        )

    lib_scal()

    # Compare results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: scal(y.torch_tensor(), x.torch_tensor(), alpha), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_scal(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
        
    check_error(LIBINFINIOP.infiniopDestroyScalDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")