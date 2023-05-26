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
