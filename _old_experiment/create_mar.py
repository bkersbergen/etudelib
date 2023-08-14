import subprocess
import os
import urllib.request

import torch

from torchvision import models

PROJECT_ID = "bolcom-pro-reco-analytics-fcc"
REGION="europe-west4"
BUCKET_URI = "gs://bolcom-pro-reco-analytics-fcc-shared/barrie_etude/trained"


def exec(command):
    print(command)
    process = subprocess.Popen(command, shell=True)
    process.wait()  # Wait for the command to finish

# Create a local directory for model artifacts
model_path = "model123"

import shutil
shutil.rmtree(model_path, ignore_errors=True)
os.makedirs(model_path)

model_name = "resnet-18-custom-handler"
model_file_jit = f"{model_path}/{model_name}.pt"
model_file_onnx = f"{model_path}/{model_name}.onnx"


# Use scripted mode to save the PyTorch model locally
model = models.resnet18(pretrained=True)
script_module = torch.jit.script(model)
script_module.save(model_file_jit)
from torch.autograd import Variable
dummy_input = Variable(torch.randn(1, 3, 224, 224))

torch.onnx.export(script_module,
                  dummy_input,
                  model_file_onnx)


hander_file = f"{model_path}/custom_handler.py"

code_string = '''\
import torch
import torch.nn.functional as F
from ts.torch_handler.image_classifier import ImageClassifier
from ts.utils.util import map_class_to_label


class CustomImageClassifier(ImageClassifier):

    # Only return the top 3 predictions
    topk = 3

    def postprocess(self, data):
        ps = F.softmax(data, dim=1)
        probs, classes = torch.topk(ps, self.topk, dim=1)
        probs = probs.tolist()
        classes = classes.tolist()
        return map_class_to_label(probs, self.mapping, classes)
'''

# Write the code string to the file
with open(hander_file, 'w') as file:
    file.write(code_string)

index_to_name_file = f"{model_path}/index_to_name.json"

urllib.request.urlretrieve(
    "https://github.com/pytorch/serve/raw/master/examples/image_classifier/index_to_name.json",
    index_to_name_file,
)

os.environ["PATH"] = f'{os.environ.get("PATH")}:~/.local/bin'

exec(f"torch-model-archiver -f \
        --model-name model \
        --version 1.0  \
        --serialized-file {model_file_jit} \
        --handler {hander_file} \
        --extra-files {index_to_name_file} \
        --export-path {model_path}")



MODEL_URI = f"{BUCKET_URI}/{model_name}"

exec(f'gsutil -m  rm -r {MODEL_URI}')
exec(f'gsutil -m cp -r {model_path} {MODEL_URI}')
exec(f'gsutil ls -al {MODEL_URI}')



