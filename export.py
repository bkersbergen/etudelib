import os.path
from enum import Enum
from pathlib import Path
import logging
import sys
import re
import os
import shutil
import torch
from pytorch_lightning import LightningModule
import onnxruntime as ort

from model_archiver.model_packaging import generate_model_archive

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


def load_jit_model(path_to_model: str, device: str):
    print('loading eager model from:' + path_to_model)
    model = torch.jit.load(path_to_model, map_location=device).to(device)
    return model


def load_onnx_session(onnx_model_path: str, device: str):
    if device == 'cuda':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    ort_sess = ort.InferenceSession(onnx_model_path, providers=providers)
    return ort_sess


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
    assert not isinstance(model, LightningModule), 'eager_model_on_cpu is not of <class "nn.Module">'
    if export_mode == ExportMode.EAGER:
        export_path = str(export_root / "model.pt.eager")
        print('export_path:' + export_path)
        torch.save(model, export_path)
        return export_path
    elif export_mode == ExportMode.JIT:
        export_path = str(export_root / "model.pt.jit")
        print('export_path:' + export_path)
        torch.jit.save(model, export_path)
        return export_path
    elif export_mode == ExportMode.ONNX:
        export_path = os.path.join(export_root, "model.onnx")
        print('export_path:' + export_path)
        torch.onnx.export(
            model,
            model_input,
            export_path,
            input_names=['item_id_list', 'max_seq_length'],  # the model's input names
            output_names=['output'],  # the model's output names
        )
        return export_path


class TorchServeExporter:

    def __init__(self):
        pass

    @staticmethod
    def export_mar_file(model_path: str, payload_path: str, output_dir: str):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filename_with_ext = os.path.basename(model_path)  # 'myfile.zip'
        filename_without_ext, file_ext = os.path.splitext(filename_with_ext)  # ('myfile', '.zip')
        rootdir = Path(__file__).parent
        handler_path = Path(rootdir, 'deploy/torch_inferencer.py')
        requirements_path = Path(rootdir, 'deploy/requirements.txt')
        sys.argv = [sys.argv[0]]  # clear the command line arguments
        sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
        sys.argv.extend(['--model-name', 'model'])  # all endpoints are called 'model' to remove complex variables in all intermediate scripts
        sys.argv.extend(['--version', '1.0'])
        sys.argv.extend(['--serialized-file', model_path])
        sys.argv.extend(['--handler', str(handler_path)])
        sys.argv.extend(['--requirements-file', str(requirements_path)])
        sys.argv.extend(['--extra-files', ','.join([payload_path])])
        sys.argv.extend(['--export-path', output_dir])
        sys.argv.extend(['--force'])

        exit_code = generate_model_archive()
        if exit_code and exit_code != 0:
            logger.error(exit_code)
            logger.error(sys.argv)
            raise RuntimeError('FAILED to create MAR')

        current_filename = output_dir + '/model.mar'
        return current_filename


