import os.path
from enum import Enum
import torch


class ExportMode(str, Enum):
    """Model export mode."""

    EAGER = "eager"
    JIT = "jit"
    ONNX = "onnx"


def export(model, model_input, export_mode: ExportMode, export_root):
    """Export model to onnx.
        Returns:
            Path: Path to the exported onnx model.
            :param model:  Model to export.
            :param model_input: input for onnx converter.
            :param export_mode: the format to export.
            :param export_root: the root folder of the exported model.
        """

    onnx_path = os.path.join(export_root, "model.onnx")
    torch.onnx.export(
        model.model,
        model_input,
        onnx_path,
        input_names=['item_id_list', 'max_seq_length'],  # the model's input names
        output_names=['output'],  # the model's output names
    )
    return onnx_path
