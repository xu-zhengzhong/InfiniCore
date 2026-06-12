import json
import os
import sys
import pprint
from pathlib import Path

# ==============================================================================
# OpTest Templates
# ==============================================================================
_TEST_FILE_TEMPLATE = r'''import sys
import os
import json
import pprint

# Path Injection
sys.path.insert(0, r"{project_root}")

import torch
import torch.nn.functional
import infinicore
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TensorInitializer,
    TestCase,
    GenericTestRunner,
)

# ==============================================================================
# Injected Configuration
# ==============================================================================
_OP_CONFIG = {op_config_json}

# ==============================================================================
# Helpers
# ==============================================================================
def _parse_dtype(dtype_str):
    """Convert string dtype to framework/torch object."""
    if hasattr(infinicore, dtype_str): return getattr(infinicore, dtype_str)
    if hasattr(torch, dtype_str): return getattr(torch, dtype_str)
    return dtype_str

def _parse_torch_dtype(dtype_str):
    if hasattr(torch, dtype_str): return getattr(torch, dtype_str)
    if hasattr(infinicore, dtype_str):
        from framework import to_torch_dtype
        return to_torch_dtype(getattr(infinicore, dtype_str))
    return dtype_str

def _parse_manual_tensor(value):
    if isinstance(value, dict) and value.get("__tensor__"):
        dtype = _parse_torch_dtype(value.get("dtype", "float32"))
        return torch.tensor(value.get("data"), dtype=dtype)
    return value

def _parse_init_mode(mode):
    if mode is None:
        return TensorInitializer.RANDOM
    return getattr(TensorInitializer, str(mode).upper(), mode)

def _dict_to_spec(spec_dict):
    """Convert JSON dict to TensorSpec object."""
    if not isinstance(spec_dict, dict): return spec_dict
    kwargs = {
        k: _parse_manual_tensor(v)
        for k, v in spec_dict.get('kwargs', {}).items()
    }
    return TensorSpec(
        shape=tuple(spec_dict['shape']),
        dtype=_parse_dtype(spec_dict['dtype']),
        name=spec_dict.get('name'),
        strides=tuple(spec_dict['strides']) if spec_dict.get('strides') else None,
        init_mode=_parse_init_mode(spec_dict.get('init_mode')),
        **kwargs
    )

def parse_test_cases():
    """Parse JSON testcases into framework TestCase objects."""
    test_cases = []
    raw_cases = _OP_CONFIG.get("testcases", [])
    for case in raw_cases:
        # 1. Parse Inputs and build name-to-index map
        input_specs = []
        name_to_index = {}
        for idx, inp in enumerate(case.get('inputs', [])):
            spec = _dict_to_spec(inp)
            input_specs.append(spec)
            if hasattr(spec, "name") and spec.name:
                name_to_index[spec.name] = idx

        # 2. Parse Kwargs
        kwargs = {}
        for k, v in case.get('kwargs', {}).items():
            # Resolve string references (e.g., "out": "a" -> "out": 0)
            if k == "out" and isinstance(v, str) and v in name_to_index:
                kwargs[k] = name_to_index[v]
            elif isinstance(v, dict) and "shape" in v:
                kwargs[k] = _dict_to_spec(v)
            else:
                kwargs[k] = v
        
        # 3. Handle explicit output spec
        output_spec = None
        if "out" in kwargs and isinstance(kwargs["out"], TensorSpec):
            output_spec = kwargs.pop("out") 
        
        # 4. Tolerance & Comparison Target
        tol_dict = case.get('tolerance', {})
        tolerance = {"atol": tol_dict.get("atol", 0), "rtol": tol_dict.get("rtol", 1e-3)}
        
        comp_target = case.get('comparison_target')
        if isinstance(comp_target, str) and comp_target in name_to_index:
            comp_target = name_to_index[comp_target]

        test_cases.append(TestCase(
            inputs=input_specs,
            kwargs=kwargs,
            output_spec=output_spec,
            comparison_target=comp_target,
            tolerance=tolerance,
            output_count=case.get('output_count', 1),
            description=case.get('description', "Dynamic Case")
        ))
    return test_cases

class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__(_OP_CONFIG.get("operator", "UnknownOp"))
        self._source_op = None
        source_module = _OP_CONFIG.get("source_module")
        if source_module:
            import importlib
            self._source_op = importlib.import_module(source_module).OpTest()

    def get_test_cases(self):
        """Returns the list of parsed test cases."""
        return parse_test_cases()

    def _resolve_kwargs(self, args, kwargs):
        """Resolves index-based 'out' arguments to actual Tensors."""
        resolved_kwargs = kwargs.copy()
        if "out" in resolved_kwargs:
            val = resolved_kwargs["out"]
            if isinstance(val, int) and 0 <= val < len(args):
                resolved_kwargs["out"] = args[val]
        return resolved_kwargs

    def torch_operator(self, *args, **kwargs):
        """PyTorch operator implementation."""
        if self._source_op is not None:
            return self._source_op.torch_operator(*args, **kwargs)
        {torch_method_body}

    def infinicore_operator(self, *args, **kwargs):
        """InfiniCore operator implementation."""
        if self._source_op is not None:
            return self._source_op.infinicore_operator(*args, **kwargs)
        {infini_method_body}

def main():
    """Execution entry point."""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()

if __name__ == "__main__":
    main()
'''

class TestGenerator:

    def __init__(self, project_root):
        self.project_root = os.path.abspath(project_root)

    def generate(self, json_list, output_dir):
        generated_files = []
        for idx, op_config in enumerate(json_list):
            op_name = op_config.get("operator", "Unknown")
            file_name = f"test_{op_name}_{idx}.py"
            file_path = os.path.join(output_dir, file_name)

            # 1. Fetch operator names
            torch_op_name = op_config.get("torch_op")
            infinicore_op_name = op_config.get("infinicore_op")

            # 2. Prepare method bodies
            # If the op name is provided, generate the return statement.
            # If it's None/null, use 'pass' to avoid syntax errors.
            make_body = lambda name, tag: (
                f"return {name}(*args, **self._resolve_kwargs(args, kwargs))" 
                if name else f"pass  # {tag} is null, skipping implementation"
            )

            torch_body = make_body(torch_op_name, "torch_op")
            infini_body = make_body(infinicore_op_name, "infinicore_op")

            # 3. Fill the template
            config_str = pprint.pformat(op_config, indent=4, width=120)
            file_content = _TEST_FILE_TEMPLATE.replace("{op_config_json}", config_str)
            file_content = file_content.replace("{project_root}", self.project_root)
            
            # Injected Method Bodies
            file_content = file_content.replace("{torch_method_body}", torch_body)
            file_content = file_content.replace("{infini_method_body}", infini_body)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(file_content)
            generated_files.append(file_path)
        return generated_files
