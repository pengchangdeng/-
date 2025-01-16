import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.nn.quantized.functional import threshold
from torch.utils.data import Dataset, DataLoader
import timm
from PIL import Image
import torchvision.transforms as transforms
from load_json import load_json
from absolute2relative import absolute2relative
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import numpy as np


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]  # 可以是多标签的形式，如 [0, 1, 0, 1, 0]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 假设有一个图片路径列表和对应的标签
# image_paths = ['E:\multi_tag\simplt\image\img.png', 'E:\multi_tag\simplt\image\img_1.png','E:\multi_tag\simplt\image\img_3.png','E:\multi_tag\simplt\image\img_4.png']  # 修改为你的图片路径
# labels = [
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 第一张图片（白色）
#     [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]# 第二张图片（黄色和白色）
# ]
# absolute2relative('merged.json','merged.json')
image_paths, labels = load_json('final.json')
# 实例化数据集和数据加载器
dataset = CustomDataset(image_paths=image_paths, labels=labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 定义自定义的 ViT 模型类
class CustomViT(nn.Module):
    def __init__(self, num_classes):
        super(CustomViT, self).__init__()
        self.model = timm.create_model('vit_tiny_patch16_224', pretrained=False)  # 使用非预训练模型
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)  # 替换分类头

    def forward(self, x):
        return self.model(x)

# 实例化模型
model = CustomViT(num_classes=19)  # 这里实例化模型，确保模型在优化器定义之前

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 适用于多标签任务
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.001)  # 学习率衰减

# 训练函数
def train(model, dataloader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")


# 验证函数
def validate(model, dataloader, criterion, device):
    # model.eval()
    # running_loss = 0.0
    # # 统计准确率
    #
    # with torch.no_grad():
    #     for inputs, labels in dataloader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #
    #         outputs = model(inputs)
    #
    #         loss = criterion(outputs, labels)
    #         running_loss += loss.item()
    #
    #
    #
    # return val_loss
    model.eval()
    running_loss = 0.0
    total_samples = 0
    correct_samples = 0

    all_labels = []
    all_predictions = []
    threshold=0.05
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Threshold outputs to get binary predictions
            predicted = (outputs > threshold).int()

            # Collect all labels and predictions for metric computation
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())

            # Example-Based Accuracy: Check if all predictions match all labels
            correct_samples += (predicted == labels).all(dim=1).sum().item()
            total_samples += labels.size(0)

    # Concatenate all labels and predictions
    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)

    # Compute metrics
    hamming_loss = np.mean(np.not_equal(all_predictions, all_labels).sum(axis=1) / all_labels.shape[1])
    example_based_accuracy = correct_samples / total_samples

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro',
                                                               zero_division=0)
    mean_average_precision = average_precision_score(all_labels, all_predictions, average='macro')

    # Compute average validation loss
    val_loss = running_loss / len(dataloader)

    return {
        'val_loss': val_loss,
        'hamming_loss': hamming_loss,
        'example_based_accuracy': example_based_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mAP': mean_average_precision
    }


# 训练过程
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, device):
    best_f1 = float('inf')

    for epoch in range(num_epochs):
        train(model, train_dataloader, criterion, optimizer, epoch, device)
        val_loss = validate(model, val_dataloader, criterion, device)

        # 学习率调整
        scheduler.step()
        f1=val_loss['f1_score']
        # 打印验证集上的指标
        print(f"Validation Loss: {val_loss['val_loss']:.4f}, "
              f"Hamming Loss: {val_loss['hamming_loss']:.4f}, "
              f"Example-Based Accuracy: {val_loss['example_based_accuracy']:.4f}, "
              f"Precision: {val_loss['precision']:.4f}, "
              f"Recall: {val_loss['recall']:.4f}, "
              f"F1 Score: {val_loss['f1_score']:.4f}, "
              f"mAP: {val_loss['mAP']:.4f}")
        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved!")


# 使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 假设有一个验证数据集
val_dataset = CustomDataset(image_paths=image_paths, labels=labels, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 训练模型
train_model(model, dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs=10, device=device)
