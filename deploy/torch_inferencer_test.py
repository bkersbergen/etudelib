from pathlib import Path

import os

from torch.utils.data import DataLoader

from deploy.torch_inferencer import TorchInferencer
from etudelib.data.synthetic.synthetic import SyntheticDataset
from etudelib.deploy.modelutil import ModelUtil
from export import TorchServeExporter

if __name__ == '__main__':
    rootdir = Path(__file__).parent.parent.parent
    BUCKET_BASE_URI='gs://bolcom-pro-reco-analytics-fcc-shared/barrie_etude/trained'
    # for C in [10_000, 100_000, 1_000_000, 5_000_000, 10_000_000, 20_000_000, 40_000_000]:
    C = 1_000_000
    t = 50
    param_source = 'bolcom'
    projectdir='.'
    # initializing the synthetic dataset takes very long for a large C value.
    train_ds = SyntheticDataset(qty_interactions=50_000,
                                qty_sessions=50_000,
                                n_items=C,
                                max_seq_length=t, param_source=param_source)
    benchmark_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    item_seq, session_length, next_item = next(iter(benchmark_loader))
    model_input = (item_seq, session_length)
    model_name='noop'
    output_path = f'{rootdir}/.docker/model_store/'
    print(f'creating model: model_name={model_name}, C={C}, max_seq_length={t}, param_source={param_source}')
    payload_path, eager_model_path, jitopt_model_path, onnx_model_path = ModelUtil.create_model(
        model_name=model_name, C=C, max_seq_length=t, param_source=param_source, model_input=model_input, projectdir=projectdir)
    directory = f'{model_name}_{param_source}_c{C}_t{t}_eager'
    location_mar_file = TorchServeExporter.export_mar_file(eager_model_path, payload_path, output_path + '/' + directory)
    # os.system(f'gsutil rm -r {BUCKET_BASE_URI}/{directory}')
    os.system(f'gsutil cp -r {location_mar_file} {BUCKET_BASE_URI}/{directory}/')

    directory = f'{model_name}_{param_source}_c{C}_t{t}_jitopt'
    location_mar_file = TorchServeExporter.export_mar_file(jitopt_model_path, payload_path,
                                                           output_path + '/' + directory)
    # os.system(f'gsutil rm -r {BUCKET_BASE_URI}/{directory}')
    os.system(f'gsutil cp -r {location_mar_file} {BUCKET_BASE_URI}/{directory}/')

    directory = f'{model_name}_{param_source}_c{C}_t{t}_onnx'
    location_mar_file = TorchServeExporter.export_mar_file(onnx_model_path, payload_path,
                                                           output_path + '/' + directory)
    # os.system(f'gsutil rm -r {BUCKET_BASE_URI}/{directory}')
    os.system(f'gsutil cp -r {location_mar_file} {BUCKET_BASE_URI}/{directory}/')


    inferencer = TorchInferencer()
    inferencer.initialize_from_file(jitopt_model_path)
    request_data = [{'body':
        {
            'instances': [{'context': [1, 2, 3]}, {'context': [2, 3, 4]}]
        }
    }]
    preprocessed = inferencer.preprocess(request_data)
    inferenced = inferencer.inference(preprocessed)
    recos = inferencer.postprocess(inferenced)
    print(recos)

    print(inferencer.handle(request_data, context=None))

    # 20230601 without vertex 'instances' and torchserve 'data'
    request_data = [{'context': [1, 2, 3]}, {'context': [2, 3, 4]}]
    print(inferencer.handle(request_data, context=None))
