import os.path
from enum import Enum
from pathlib import Path
import logging

import torch
from pytorch_lightning import LightningModule
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ExportMode(str, Enum):
    """Model export mode."""

    EAGER = "eager"
    JIT = "jit"
    ONNX = "onnx"


def load_eager_model(path_to_model: str, device: str):
    print('loading eager model from:' + path_to_model)
    model = torch.load(path_to_model, map_location=device).to(device)
    return model

def save_eager_model(model, export_root: Path):
    Path(export_root).mkdir(parents=True, exist_ok=True)
    assert not isinstance(model, LightningModule), 'eager_model_on_cpu is not of <class "nn.Module">'
    export_path = str(export_root / "model.pt.eager")
    print('export_path:' + export_path)
    torch.save(model, export_path)
    return export_path


def load_jit_model(path_to_model: str, device: str):
    print('loading eager model from:' + path_to_model)
    model = torch.jit.load(path_to_model, map_location=device).to(device)
    return model
def save_jit_model(jit_model, export_root: Path):
    Path(export_root).mkdir(parents=True, exist_ok=True)
    assert not isinstance(jit_model, LightningModule), 'eager_model_on_cpu is not of <class "nn.Module">'
    export_path = str(export_root / "model.pt.jit")
    print('export_path:' + export_path)
    torch.jit.save(jit_model, export_path)
    return export_path


def load_onnx_session(onnx_model_path: str, device: str):
    if device == 'cuda':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    ort_sess = ort.InferenceSession(onnx_model_path, providers=providers)
    return ort_sess

def save_onnx_model(eager_model, export_root: Path, model_input):
    Path(export_root).mkdir(parents=True, exist_ok=True)
    assert not isinstance(eager_model, LightningModule), 'eager_model_on_cpu is not of <class "nn.Module">'
    export_path = str(export_root / "model.pt.onnx")
    print('export_path:' + export_path)
    torch.onnx.export(
        eager_model,
        model_input,
        export_path,
        input_names=['item_id_list', 'max_seq_length'],  # the model's input names
        output_names=['output'],  # the model's output names
    )
    return export_path


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
        torch.jit.save(model, export_path)
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
