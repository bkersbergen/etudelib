from .modelutil import ModelUtil

from etudelib.deploy.export import TorchServeExporter


def test_create_mar():
    model_name = 'core'
    output_path = '/tmp'
    payload_path, eager_model_path, jitopt_model_path, onnx_model_path = ModelUtil.create_model('core')
    TorchServeExporter.export_mar_file(model_name, onnx_model_path, payload_path, output_path)


if __name__ == '__main__':
    test_create_mar()
