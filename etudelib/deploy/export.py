import os.path
from enum import Enum
from pathlib import Path
import logging

import torch
from pytorch_lightning import LightningModule

logger = logging.getLogger(__name__)


class ExportMode(str, Enum):
    """Model export mode."""

    EAGER = "eager"
    JIT = "jit"
    ONNX = "onnx"


def load_torch_model(path_to_model: str, device: str):
    print('loading model from:' + path_to_model)
    # return torch.load(path_to_model, map_location=device).to(device)
    return torch.load(path_to_model)


def export(model, model_input, export_mode: ExportMode, export_root: Path):
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
        assert not isinstance(model, LightningModule), 'eager_model_on_cpu is not of <class "nn.Module">'
        export_path = str(export_root / "model.pt.eager")
        print('export_path:' + export_path)
        torch.save(model, export_path)
        return export_path
    elif export_mode == ExportMode.JIT:
        assert not isinstance(model, LightningModule), 'eager_model_on_cpu is not of <class "nn.Module">'
        export_path = str(export_root / "model.pt.jit")
        torch.save(model, export_path)
        return export_path
    elif export_mode == ExportMode.ONNX:
        onnx_path = os.path.join(export_root, "model.onnx")
        torch.onnx.export(
            model,
            model_input,
            onnx_path,
            input_names=['item_id_list', 'max_seq_length'],  # the model's input names
            output_names=['output'],  # the model's output names
        )
        return onnx_path
