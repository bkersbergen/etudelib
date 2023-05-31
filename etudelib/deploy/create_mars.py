import os
from pathlib import Path

from torch.utils.data import DataLoader

from etudelib.data.synthetic.synthetic import SyntheticDataset
from etudelib.deploy.modelutil import ModelUtil

from etudelib.deploy.export import TorchServeExporter
import subprocess


def create_mars():
    rootdir = Path(__file__).parent.parent.parent
    BUCKET_BASE_URI='gs://bolcom-pro-reco-analytics-fcc-shared/barrie_etude/trained'
    # for C in [10_000, 100_000, 1_000_000, 5_000_000, 10_000_000, 20_000_000, 40_000_000]:
    for C in [1_000_000]:
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
        # for model_name in ['core', 'gcsan', 'gru4rec', 'lightsans', 'narm', 'noop', 'repeatnet', 'sasrec', 'sine', 'srgnn',
        #            'stamp']:
        for model_name in ['noop']:
            output_path = f'{rootdir}/.docker/model_store/'
            print(f'creating model: model_name={model_name}, C={C}, max_seq_length={t}, param_source={param_source}')
            payload_path, eager_model_path, jitopt_model_path, onnx_model_path = ModelUtil.create_model(
                model_name=model_name, C=C, max_seq_length=t, param_source=param_source, model_input=model_input)
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


            #
            # # docker_build_push(eager_model_path)
            # location_mar_file = TorchServeExporter.export_mar_file(jitopt_model_path, payload_path,
            #                                                        output_path + '/' + f'${model_name}_${param_source}_c${C}_t${t}_jitopt')
            # # docker_build_push(jitopt_model_path)
            # location_mar_file = TorchServeExporter.export_mar_file(jitopt_model_path, payload_path,
            #                                                        output_path + '/' + f'${model_name}_${param_source}_c${C}_t${t}_onnx')
            # # docker_build_push(onnx_model_path)
            #
            # exec(f'gsutil -m  rm -r {MODEL_URI}')
            # exec(f'gsutil -m cp -r {model_path} {MODEL_URI}')
            # exec(f'gsutil ls -al {MODEL_URI}')


def docker_build_push(model_path):
    filename_with_ext = os.path.basename(model_path)  # 'myfile.zip'
    filename_without_ext, file_ext = os.path.splitext(filename_with_ext)  # ('myfile', '.zip')
    rootdir = Path(__file__).parent.parent.parent
    task = ['make', 'model_build', f'MARFILE_WO_EXT={filename_without_ext}']
    print(rootdir)
    p = subprocess.Popen(task, cwd=rootdir)
    p.wait()


if __name__ == '__main__':
    create_mars()
