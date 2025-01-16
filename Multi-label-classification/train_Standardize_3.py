import argparse
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


# Focal Loss 实现
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 输入通过sigmoid函数得到预测概率
        inputs = torch.sigmoid(inputs)

        # 计算每个类别的交叉熵
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)

        # 计算p_t = sigmoid(x) * y + (1 - sigmoid(x)) * (1 - y)
        p_t = inputs * targets + (1 - inputs) * (1 - targets)

        # 计算Focal Loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * BCE_loss

        # 根据reduction方式计算最终损失
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


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
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 自定义 ViT 模型
class CustomViT(nn.Module):
    def __init__(self, num_classes):
        super(CustomViT, self).__init__()
        self.model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
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

        if scaler:
            with autocast():
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
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predicted = (outputs > threshold).int()
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
            correct_samples += (predicted == labels).all(dim=1).sum().item()
            total_samples += labels.size(0)

    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)

    hamming_loss = np.mean(np.not_equal(all_predictions, all_labels).sum(axis=1) / all_labels.shape[1])
    example_based_accuracy = correct_samples / total_samples

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro',
                                                               zero_division=0)
    mean_average_precision = average_precision_score(all_labels, all_predictions, average='macro')

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


# 主函数
def main(opt):
    # 加载数据集
    dataset = CustomDataset(image_paths=opt.image_paths, labels=opt.labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    # 初始化模型
    model = CustomViT(num_classes=19)
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = FocalLoss(alpha=opt.alpha, gamma=opt.gamma)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # 混合精度训练
    scaler = GradScaler()

    # 训练模型
    best_f1 = -float('inf')
    for epoch in range(opt.num_epochs):
        train(model, dataloader, criterion, optimizer, epoch, device, scaler)
        val_loss = validate(model, dataloader, criterion, device)
        scheduler.step(val_loss['val_loss'])

        # print(f"Epoch {epoch}: F1 Score {val_loss['f1_score']:.4f}, Best F1 {best_f1:.4f}")
        # 打印验证集上的指标
        print(f"Epoch {epoch}: Validation Loss: {val_loss['val_loss']:.4f}, "
              f"Hamming Loss: {val_loss['hamming_loss']:.4f}, "
              f"Example-Based Accuracy: {val_loss['example_based_accuracy']:.4f}, "
              f"Precision: {val_loss['precision']:.4f}, "
              f"Recall: {val_loss['recall']:.4f}, "
              f"F1 Score: {val_loss['f1_score']:.4f}, "
              f"mAP: {val_loss['mAP']:.4f}")


        if val_loss['f1_score'] > best_f1:
            best_f1 = val_loss['f1_score']
            save_path = os.path.join(opt.save_dir, 'best_model_data_pro.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")


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

    opt = parser.parse_args()
    print(opt)

    json_data_path = r"E:\multi_g\simplt_1\simplt\data\json_data\final_output.json"
    jpg_base_path = r'E:\multi_g\simplt_1\simplt\data\jpg_data\\'
    # 加载图片路径和标签
    image_paths, labels = load_json(json_data_path, jpg_base_path)
    opt.image_paths = image_paths
    opt.labels = labels

    main(opt)


