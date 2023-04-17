from .modelutil import ModelUtil

from etudelib.deploy.export import TorchServeExporter


def test_create_mar():
    for model_name in ['core', 'noop']:
        C=100000
        output_path = '../.docker/model_store/'
        payload_path, eager_model_path, jitopt_model_path, onnx_model_path = ModelUtil.create_model(model_name=model_name, C=C)
        TorchServeExporter.export_mar_file(eager_model_path, payload_path, output_path)
        TorchServeExporter.export_mar_file(jitopt_model_path, payload_path, output_path)
        TorchServeExporter.export_mar_file(onnx_model_path, payload_path, output_path)


if __name__ == '__main__':
    test_create_mar()
