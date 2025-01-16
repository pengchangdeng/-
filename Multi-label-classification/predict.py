import torch
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
import torch.nn as nn

color_labels = {
        "red": 0,
        "green": 1,
        "blue": 2,
        "yellow": 3,
        "orange": 4,
        "purple": 5,
        "pink": 6,
        "black": 7,
        "white": 8,
        "gray": 9,
        "box_car": 10,
        "pickup_car": 11,
        "bus":12,
        "truck":13,
        "work_car_head":14,
        "work_car":15,
        "car":16,
        "suv":17,
        "fire_truck":18
    }

# CustomViT 类定义
class CustomViT(nn.Module):
    def __init__(self, num_classes):
        super(CustomViT, self).__init__()
        self.model = timm.create_model('vit_tiny_patch16_224', pretrained=True)  # 使用非预训练模型
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)  # 替换分类头
        # 添加 Dropout 层
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return self.model(x)


# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载训练好的模型权重
model = CustomViT(num_classes=19)
model.load_state_dict(torch.load(r'E:\multi_g\simplt_1\simplt\model_file\gao\best_model1111111.pth'))
model.to(device)  # 将模型转移到 GPU

model.eval()

# 图像预处理
# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 适量使用
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 定义预测函数
def predict_image(model, image_path, threshold=0.2):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # 将输入图片也转移到 GPU

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs).squeeze(0)  # 获取每个标签的概率
        labels = (probs > threshold).float()  # 转换为二值标签

    return probs.cpu().numpy(), labels.cpu().numpy()  # 将结果转回 CPU 进行后续处理

# 示例：对单张图片进行预测
image_path = r'E:\multi_g\simplt_1\simplt\data\jpg_data\car\accent_0_frame_0096_cropped_0.jpg'  # 替换为实际图片路径
probs, labels = predict_image(model, image_path)

print("Predicted Probabilities:", [f"{prob:.3f}" for prob in probs])
print("Predicted Labels (Binary):", labels)

# 获取值为 1 的标签索引
indices_of_ones = np.where(labels == 1)[0]
print("Indices of 1s:", indices_of_ones)

# 使用 color_labels 字典进行反向查找，获取对应的颜色名称
# 使用 color_labels 字典进行反向查找，获取对应的颜色名称及其概率
predicted_colors_with_probs = {color: probs[label] for color, label in color_labels.items() if label in indices_of_ones}

# 输出结果，显示颜色名称和对应的概率（保留三位小数）
print("Predicted Colors with Probabilities:")
for color, prob in predicted_colors_with_probs.items():
    print(f"{color}: {prob:.3f}")

