from torch.utils.data import DataLoader

from etudelib.data.synthetic.synthetic import SyntheticDataset
from etudelib.deploy.modelutil import ModelUtil

from etudelib.deploy.export import TorchServeExporter


def create_mars():

    for C in [1_000, 10_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 20_000_000]:
        t = 50
        param_source = 'bolcom'
        # initializing the synthetic dataset takes very long for a large C value.
        train_ds = SyntheticDataset(qty_interactions=50_000,
                                    qty_sessions=50_000,
                                    n_items=C,
                                    max_seq_length=t, param_source=param_source)
        benchmark_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
        item_seq, session_length, next_item = next(iter(benchmark_loader))
        model_input = (item_seq, session_length)
        for model_name in ['core', 'gcsan', 'gru4rec', 'lightsans', 'narm', 'noop', 'repeatnet', 'sasrec', 'sine', 'srgnn',
                   'stamp']:
            output_path = '../.docker/model_store/'
            print(f'creating model: model_name={model_name}, C={C}, max_seq_length={t}, param_source={param_source}')
            payload_path, eager_model_path, jitopt_model_path, onnx_model_path = ModelUtil.create_model(model_name=model_name, C=C, max_seq_length=t, param_source=param_source, model_input=model_input)
            TorchServeExporter.export_mar_file(eager_model_path, payload_path, output_path)
            TorchServeExporter.export_mar_file(jitopt_model_path, payload_path, output_path)
            TorchServeExporter.export_mar_file(onnx_model_path, payload_path, output_path)


if __name__ == '__main__':
    create_mars()
