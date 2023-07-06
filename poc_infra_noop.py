# import os
# from pathlib import Path
#
# from torch.utils.data import DataLoader
#
# from etudelib.data.synthetic.synthetic import SyntheticDataset
# from etudelib.deploy.modelutil import ModelUtil
#
# from export import TorchServeExporter
#
#
# def create_mars():
#     C=1_000_000
#     t = 50
#     model_name='noop'
#     param_source = 'bolcom'
#     rootdir = Path(__file__).parent
#     base_filename = f'{model_name}_{param_source}_c{C}_t{t}'
#     project_path = f'{rootdir}/projects/{base_filename}'
#     BUCKET_BASE_URI = 'gs://bolcom-pro-reco-analytics-fcc-shared/barrie_etude/trained'
#     # initializing the synthetic dataset takes very long for a large C value.
#     train_ds = SyntheticDataset(qty_interactions=50_000,
#                                 qty_sessions=50_000,
#                                 n_items=C,
#                                 max_seq_length=t, param_source=param_source)
#     benchmark_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
#     item_seq, session_length, next_item = next(iter(benchmark_loader))
#     model_input = (item_seq, session_length)
#
#     print(f'creating model: model_name={model_name}, C={C}, max_seq_length={t}, param_source={param_source}')
#     payload_path, eager_model_path, jitopt_model_path, onnx_model_path = ModelUtil.create_model(
#         model_name=model_name, C=C, max_seq_length=t, param_source=param_source, model_input=model_input, projectdir=project_path)
#     directory = f'{base_filename}_eager'
#     location_mar_file = TorchServeExporter.export_mar_file(eager_model_path, payload_path, project_path + '/' + directory)
#     os.system(f'gsutil cp -r {location_mar_file} {BUCKET_BASE_URI}/{directory}/')
#
#     directory = f'{base_filename}_jitopt'
#     location_mar_file = TorchServeExporter.export_mar_file(jitopt_model_path, payload_path,
#                                                            project_path + '/' + directory)
#     os.system(f'gsutil cp -r {location_mar_file} {BUCKET_BASE_URI}/{directory}/')
#     os.system(f'cp {location_mar_file} {rootdir}/.docker/torchserve/models/')
#
#
#
#
# if __name__ == '__main__':
#     create_mars()
