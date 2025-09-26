import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
import torch.nn as nn

# 1. 加载 FashionMNIST 数据集
training_data = datasets.FashionMNIST(
    root="data",          # 数据存储路径
    train=True,           # 加载训练集
    download=True,        # 如果数据不存在则下载
    transform=ToTensor()  # 将图像转换为张量
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,          # 加载测试集
    download=True,
    transform=ToTensor()
)

# 2. 可视化数据集中的样本
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()  # 随机选择一个样本
    img, label = training_data[sample_idx]  # 获取图像和标签
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])  # 显示类别名称
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")  # 显示灰度图像
plt.show()

# 3. 创建自定义数据集
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        """
        初始化函数
        :param annotations_file: 标签文件的路径（CSV 格式）
        :param img_dir: 图像存储的目录
        :param transform: 图像预处理函数
        :param target_transform: 标签预处理函数
        """
        self.img_labels = pd.read_csv(annotations_file)  # 读取标签文件
        self.img_dir = img_dir  # 图像目录
        self.transform = transform  # 图像变换
        self.target_transform = target_transform  # 标签变换

    def __len__(self):
        """返回数据集的大小"""
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        根据索引返回一个样本
        :param idx: 样本索引
        :return: 图像和标签
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])  # 获取图像路径
        image = read_image(img_path)  # 读取图像
        label = self.img_labels.iloc[idx, 1]  # 获取标签
        if self.transform:
            image = self.transform(image)  # 对图像进行变换
        if self.target_transform:
            label = self.target_transform(label)  # 对标签进行变换
        return image, label

# 4. 使用 DataLoader 加载数据
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)  # 训练集 DataLoader
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)       # 测试集 DataLoader

# 遍历 DataLoader
train_features, train_labels = next(iter(train_dataloader))  # 获取一个批次的数据
print(f"Feature batch shape: {train_features.size()}")  # 打印特征张量的形状
print(f"Labels batch shape: {train_labels.size()}")     # 打印标签张量的形状

# 可视化一个样本
img = train_features[0].squeeze()  # 去除批次维度
label = train_labels[0]            # 获取标签
plt.imshow(img, cmap="gray")       # 显示图像
plt.show()
print(f"Label: {label}")           # 打印标签

# 5. 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 6. 初始化模型、损失函数和优化器
device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# 7. 定义训练和测试函数
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 计算预测误差
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# 8. 训练模型
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# 9. 保存模型
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# 10. 加载模型并测试
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')