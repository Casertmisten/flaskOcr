import torch

from models.Ocr_V1 import OcrNetV1


# 只加载权重，加载到cpu上
def load_trained_model(model_path):
    model = OcrNetV1()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model
