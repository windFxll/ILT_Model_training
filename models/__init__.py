# from .Model import UNetFull, UNetEdge, UNetEdge_64, UNetEdgePreserve

# __all__ = ["UNetFull", "UNetEdge", "UNetEdge_64", "UNetEdgePreserve"]

import os
import importlib

model_dir = os.path.dirname(__file__)

for file in os.listdir(model_dir):
    if file.endswith(".py") and file not in ["__init__.py", "registry.py"]:
        module_name = f"models.{file[:-3]}"
        importlib.import_module(module_name)