from torchvision import transforms


# 1.转化为灰度图 2.转化为tensor 3.改变形状
def preprocess_image(image, size=(28, 28)):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    image = image.convert('L')  # 转换为灰度图
    image = transform(image)
    return image.unsqueeze(0)  # 增加 batch 维度，以符合输入标准
