import logging
from timeit import default_timer as timer

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class Static(BaseHandler):

    def __init__(self):
        pass

    def initialize(self, context):
        #  load the model
        logger.info('Static.initialize(): {}'.format(context))
        properties = context.system_properties
        logger.info("properties: {}".format(properties))

    def handle(self, data, context):
        self.context = context
        t0 = timer()
        t1 = timer()
        items = list(range(1, 21))
        t2 = timer()
        t3 = timer()
        preprocess_time_ms = (t1 - t0) * 1000
        inference_time_ms = (t2 - t1) * 1000
        postprocess_time_ms = (t3 - t2) * 1000
        output = [{'items': items, 'nf': {'preprocess_ms': preprocess_time_ms,
                                                       'inference_ms': inference_time_ms,
                                                       'postprocess_ms': postprocess_time_ms,
                                                       'model': 'static_json',
                                                       'device': 'cpu',
                                      }}]
        return output

#
# if __name__ == '__main__':
#     model_name = 'core'
#     C = 1000000
#     max_seq_length = 50
#     dataset_name = 'synthetic'
#     rootdir = Path(__file__).parent.parent.parent.parent
#
#     projectdir = Path(rootdir, 'projects/benchmark')
#     configure_logger(level='INFO')
#
#     config_path = os.path.join(rootdir, f"etudelib/models/{model_name}/config.yaml".lower())
#     config = OmegaConf.load(config_path)
#
#     config['dataset'] = {}
#     config['dataset']['n_items'] = C
#     config['dataset']['max_seq_length'] = max_seq_length
#
#     module = import_module(f"etudelib.models.{config.model.name}.lightning_model".lower())
#     model = getattr(module, f"{config.model.name}Lightning")(config)
#
#     eager_model = model.get_backbone()
#
#     eager_model = TopKDecorator(eager_model, topk=21)
#     eager_model.eval()
#
#     base_filename = f'{model_name}_{dataset_name}_{C}_{max_seq_length}'
#
#     payload = {'max_seq_length': max_seq_length,
#                'C': C,
#                'idx2item': [i for i in range(C)]
#                }
#     torch.save(payload, str(projectdir / f'{base_filename}_payload.torch'))
#
#     eager_model_file = str(projectdir / f'{base_filename}_eager.pth')
#     torch.save(eager_model, eager_model_file)
#
#     inferencer = TorchInferencer()
#     inferencer.initialize_from_file(eager_model_file)
#     request_data = [{'body':
#         {
#             'instances': [{'context': [1, 2, 3]}, {'context': [2, 3, 4]}]
#         }
#     }]
#     preprocessed = inferencer.preprocess(request_data)
#     inferenced = inferencer.inference(preprocessed)
#     recos = inferencer.postprocess(inferenced)
#
#     recos = inferencer.handle(request_data, context=None)
#     print(recos)
