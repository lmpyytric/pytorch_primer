import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data=datasets.FashionMNIST(root="data",train=True,download=True,transform=ToTensor())
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
bath_size=64
train_dataloader=DataLoader(train_data,bath_size)
test_dataloader=DataLoader(test_data,bath_size)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.flatten=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    def forward(self,x):
        x=self.flatten(x)
        x=self.linear_relu_stack(x)
        return x

model=NeuralNetwork()
model.cuda()
print(model)
loss_fn = nn.CrossEntropyLoss()  # 损失函数设置
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 学习率设置
epochs = 5
for images,labels in train_dataloader:
    images.cuda()
    labels.cuda()
    pred=model(images)
    loss=loss_fn(pred,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 测试集大小
size = len(test_dataloader.dataset)
    # 测试集的batch数量
num_batches = len(test_dataloader)
model.eval()
# 记录loss和准确率
test_loss, correct = 0, 0
# 梯度截断
with torch.no_grad():
    for images, labels in test_dataloader:  # 遍历batch
        # 加载到device
        images.cuda()
        labels.cuda()
        # 输入数据到模型里得到输出
        pred = model(images)
        # 累加loss
        test_loss += loss_fn(pred, labels).item()
        # 累加正确率
correct += (pred.argmax(1) == labels).sum().item()
# 计算平均loss和准确率
test_loss /= num_batches
correct /= size
print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
