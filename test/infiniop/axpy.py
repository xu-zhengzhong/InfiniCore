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
    ((13,), None, None, 2.5),
    ((13,), (10,), (10,), 2.5),
    ((5632,), None, None, 2.5),
    ((5632,), (5,), (5,), 2.5),
    ((16,), (4,), (4,), 2.5),
    ((5632,), (32,), (32,), 2.5),
]

# Data types used for testing
_TENSOR_DTYPES = [
    InfiniDtype.F32,
    # InfiniDtype.F64,
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.F64: {"atol": 1e-15, "rtol": 1e-15},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def axpy(x, y, alpha):
    y.add_(x * alpha)


def test(
    handle,
    device,
    shape,
    x_stride=None,
    y_stride=None,
    alpha_val=2.5,
    dtype=torch.float16,
    sync=None,
):
    if dtype == InfiniDtype.F32:
        alpha = ctypes.c_float(alpha_val)
    elif dtype == InfiniDtype.F64:
        alpha = ctypes.c_double(alpha_val)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    x = TestTensor(shape, x_stride, dtype, device)
    y = TestTensor(shape, y_stride, dtype, device)

    # In-place broadcast target is structurally invalid for axpy
    if x.is_broadcast():
        return

    print(
        f"Testing Axpy on {InfiniDeviceNames[device]} with shape:{shape} x_stride:{x_stride} "
        f"y_stride:{y_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    # Compute PyTorch reference
    axpy(x.torch_tensor(), y.torch_tensor(), alpha_val)

    if sync is not None:
        sync()

    # Create Descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateAxpyDescriptor(
            handle,
            ctypes.byref(descriptor),
            x.descriptor,
            y.descriptor
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x.destroy_desc()
    y.destroy_desc()

    # Allocate Workspace
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetAxpyWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    # Execute C library op
    def lib_axpy():
        check_error(
            LIBINFINIOP.infiniopAxpy(
                descriptor,
                workspace.data(),
                workspace.size(),
                ctypes.byref(alpha),
                x.data(),
                y.data(),
                None,
            )
        )

    lib_axpy()

    # Compare results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: axpy(x.torch_tensor(), y.torch_tensor(), alpha), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_axpy(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
        
    check_error(LIBINFINIOP.infiniopDestroyAxpyDescriptor(descriptor))


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