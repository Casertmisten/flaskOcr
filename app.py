import base64
import io
import torch
from flask import Flask, request

from PIL import Image
from methods.prepreocess import preprocess_image
from methods.Load_Models import load_trained_model


app = Flask(__name__)

# 加载模型
model = load_trained_model('models/trained_cor_model_weight_v1.pth')


@app.route('/predict', methods=['POST'])
def predict():
    # 获取前端发送的base64
    image_data = request.json.get('image')

    # 检查是否收到了有效的图像数据
    if not image_data:
        return {"error": "No image data received"}, 400

    # 解码 base64 字符串
    try:
        image_data = base64.b64decode(image_data)
    except Exception as e:
        return {"error": f"Failed to decode base64 data: {str(e)}"}, 400

    # 将图像数据转换为 PIL 图像对象
    try:
        image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        return {"error": f"Failed to create PIL image: {str(e)}"}, 400

    image = preprocess_image(image)
    # 使用 OCR 模型进行识别
    with torch.no_grad():
        output = model(image.view(-1, 28 * 28))
        predicted_class = torch.argmax(output, dim=1).item()

    predict_word = str(predicted_class)
    words_list = [
        {"words": predict_word}
    ]

    response = {
        "words_result": words_list,
        "words_result_num": 1,
        "log_id": 0,
    }
    # 打印预测结果
    print(f"The predicted number is: {predicted_class}")
    # 返回预测结果
    return response, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
