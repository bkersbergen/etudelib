import os.path
from enum import Enum
from pathlib import Path
from typing import Dict
import logging

import torch
from pytorch_lightning import LightningModule
from torch.nn import Module

logger = logging.getLogger(__name__)


class ExportMode(str, Enum):
    """Model export mode."""

    EAGER = "eager"
    JIT = "jit"
    ONNX = "onnx"


def save_eager_model(eager_model_on_cpu: Module, export_root: Path):
    assert not isinstance(eager_model_on_cpu, LightningModule), 'eager_model_on_cpu is not of <class "nn.Module">'
    export_root.mkdir(parents=True, exist_ok=True)
    export_path = export_root / "model.pt"
    torch.save(eager_model_on_cpu, export_path)
    return export_path


def load_eager_model(path_to_eager_model: Path, device: str):
    return torch.load(path_to_eager_model, map_location=device).to(device)


def export(eager_model, model_input, export_mode: ExportMode, export_root):
    """Export model to onnx.
        Returns:
            Path: Path to the exported onnx model.
            :param model:  Model to export.
            :param model_input: a single request input as an example for the optimizer.
            :param export_mode: the format to export.
            :param export_root: the root folder of the exported model.
        """
    Path(export_root).mkdir(parents=True, exist_ok=True)
    if export_mode == ExportMode.EAGER:
        save_eager_model(eager_model, export_root)
        raise RuntimeError('todo barrie needs to implement export_mode.EAGER')
        pass
    elif export_mode == ExportMode.JIT:
        # pytorch compiled
        jit_path = os.path.join(export_root, "model.ptc")
        raise RuntimeError('todo barrie needs to implement export_mode.JIT')
        pass
    elif export_mode == ExportMode.ONNX:
        onnx_path = os.path.join(export_root, "model.onnx")
        torch.onnx.export(
            eager_model,
            model_input,
            onnx_path,
            input_names=['item_id_list', 'max_seq_length'],  # the model's input names
            output_names=['output'],  # the model's output names
        )
        return onnx_path
