import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from PIL import Image
import torchvision.transforms as transforms
from load_json import load_json
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import os
import argparse

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
    transforms.RandomHorizontalFlip(),  # 适量使用
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 自定义的 ViT 模型类
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


# 训练函数
def train(model, dataloader, criterion, optimizer, epoch, device, scaler=None):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 混合精度训练
        if scaler:
            with autocast():  # 自动混合精度
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")


# 验证函数
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    correct_samples = 0

    all_labels = []
    all_predictions = []
    threshold = 0.05
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
def train_model(opt):
    # 加载图片路径和标签
    dataset = CustomDataset(image_paths=image_paths, labels=labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    # 实例化模型
    model = CustomViT(num_classes=19)
    model.to(opt.device)

    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()  # 适用于多标签任务
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)  # 学习率调度

    # 混合精度训练
    scaler = GradScaler()

    # 实例化验证数据集
    val_dataset = CustomDataset(image_paths=image_paths, labels=labels, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    best_f1 = float('-inf')  # 应该用负无穷开始，以便 F1 总是能增加

    for epoch in range(opt.num_epochs):
        train(model, dataloader, criterion, optimizer, epoch, opt.device, scaler)
        val_loss = validate(model, val_dataloader, criterion, opt.device)

        # 学习率调整
        scheduler.step(val_loss['val_loss'])
        f1 = val_loss['f1_score']

        print(f"Epoch {epoch}: Validation Loss: {val_loss['val_loss']:.4f}, "
              f"Hamming Loss: {val_loss['hamming_loss']:.4f}, "
              f"Example-Based Accuracy: {val_loss['example_based_accuracy']:.4f}, "
              f"Precision: {val_loss['precision']:.4f}, "
              f"Recall: {val_loss['recall']:.4f}, "
              f"F1 Score: {val_loss['f1_score']:.4f}, "
              f"mAP: {val_loss['mAP']:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(opt.save_dir, 'best_model_normal.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print(f"F1 improved from {best_f1} to {f1}, saving model to {save_path}...")

            model.cpu()
            torch.save(model.state_dict(), save_path)
            print("Best model saved!")
            model.to(opt.device)


# 主函数入口
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda', help='device for training')
    parser.add_argument('--alpha', type=float, default=0.25, help='alpha for focal loss')
    parser.add_argument('--gamma', type=float, default=2.0, help='gamma for focal loss')
    parser.add_argument('--save_dir', type=str, default='output', help='directory to save the model')
    # parser.add_argument('--json_data_path', type=str, required=True, help='path to json data')
    # parser.add_argument('--jpg_base_path', type=str, required=True, help='path to base jpg images')

    opt = parser.parse_args()
    print(opt)

    json_data_path = r"E:\multi_g\simplt_1\simplt\data\json_data\final.json"
    jpg_base_path = r'E:\multi_g\simplt_1\simplt\data\jpg_data\\'
    # 加载图片路径和标签
    image_paths, labels = load_json(json_data_path, jpg_base_path)

    opt.image_paths = image_paths
    opt.labels = labels

    train_model(opt)

