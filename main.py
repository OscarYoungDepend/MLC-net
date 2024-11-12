import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import normalize,StandardScaler
from myloss import multilabel_categorical_crossentropy
from torch.utils.data import random_split
def extract_i(Y,index):
    extract_e=Y[index]
    mask=torch.ones(Y.size(0),dtype=torch.bool,device=Y.device)
    mask[index]=False
    rem=Y[mask]
    return extract_e,rem
# 加载数据
data = sio.loadmat('./networkdata/X_t.mat')['X_t'].transpose()  # 复数信号

labels = sio.loadmat('./networkdata/X_label.mat')['XX'].transpose() # 标签
index=list(range(45000,92744,10))
#print(torch.cuda.is_available())
#testdata,traindata=extract_i(data,index)
device = torch.device("cuda:0")
# 数据预处理
real_part = np.real(data)
imag_part = np.imag(data)
#real_part=StandardScaler(real_part,axis=1)
#imag_part=StandardScaler(imag_part,axis=1)
real_part = np.expand_dims(real_part, axis=2)
imag_part = np.expand_dims(imag_part, axis=2)

real_part = torch.tensor(real_part, dtype=torch.float32).cuda()
imag_part = torch.tensor(imag_part, dtype=torch.float32).cuda()
labels = torch.tensor(labels, dtype=torch.float32).cuda()

# 合并实部和虚部作为输入通道
data = torch.cat((real_part, imag_part), dim=2).permute(0,2,1)
#dataset = TensorDataset(data, labels)
# 划分训练集和测试集
#train_size = int(0.9 * len(data))
test_data,train_data=extract_i(data,index)
#test_data=data[index]
#train_data=torch.tensor([item for i,item in enumerate(data) if i not in index]).cuda()
#train_data, test_data = random_split(dataset=dataset,lengths=[train_size, int(len(data))-train_size])
#train_labels, test_labels = labels[:train_size], labels[index]
#test_labels=labels[index]
#train_labels=torch.tensor([item for i,item in enumerate(labels) if i not in index]).cuda()
test_labels,train_labels=extract_i(labels,index)
print(len(train_labels),len(test_labels),len(train_data))

# 创建数据加载器
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset,batch_size=int(len(test_data)))
def evaluate_accuracy(X, y, model):
    pred = model(X).cpu()
    y=y.cpu()
    correct = sum(row.all().int().item() for row in (pred.ge(0.5) == y))
    n = y.shape[0]
    return correct / n
def magiclabel(X,t):


    top_two_indices = np.argsort(X, axis=1)[:, -2:]

    # Create a mask where values greater than 0.7 are set to True, and others to False
    mask = (X[np.arange(len(X))[:, None], top_two_indices] > t)

    # Create a new array where values greater than 0.7 are preserved, and others are set to 0
    result = np.zeros_like(X)
    result[np.arange(len(X))[:, None], top_two_indices] = X[np.arange(len(X))[:, None], top_two_indices] * mask
    X = result
    # trick1
    max_values = np.max(X, axis=1)
    all_below_threshold = np.all(X < t, axis=1)
    X[all_below_threshold, np.argmax(X, axis=1)[all_below_threshold]] = 1
    return X

def evaluate_accuracy1(X, y, t):
    top_two_indices = np.argsort(X, axis=1)[:, -2:]

    # Create a mask where values greater than 0.7 are set to True, and others to False
    mask = (X[np.arange(len(X))[:, None], top_two_indices] > t)

    # Create a new array where values greater than 0.7 are preserved, and others are set to 0
    result = np.zeros_like(X)
    result[np.arange(len(X))[:, None], top_two_indices] = X[np.arange(len(X))[:, None], top_two_indices] * mask
    X = result
    # trick1
    max_values = np.max(X, axis=1)
    all_below_threshold = np.all(X < t, axis=1)
    X[all_below_threshold, np.argmax(X, axis=1)[all_below_threshold]] = 1
    pred = torch.tensor(X)
    y = torch.tensor(y)

    correct = sum(row.all().int().item() for row in (pred.ge(t) == y))
    n = y.shape[0]
    return correct / n
class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()

# 构建神经网络模型
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiLabelClassifier, self).__init__()
        self.conv_unit = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=11, stride=1, padding=5,padding_mode='reflect'),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=11, stride=1, padding=5,padding_mode='reflect'),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.AdaptiveAvgPool1d(64),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2,padding_mode='reflect'),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2,padding_mode='reflect'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2,padding_mode='reflect'),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),

            #nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2,padding_mode='reflect'),
            #nn.LeakyReLU(),

            #nn.Conv1d(in_channels=64, out_channels=16, kernel_size=5, stride=1, padding=2,padding_mode='reflect'),
            #nn.LeakyReLU(),
            #nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1,padding_mode='reflect'),
            #nn.LeakyReLU(),
            nn.MaxPool1d(4),
            #nn.Dropout(0.1),
            #nn.Flatten(),

            nn.Dropout(0.2),
        )
        self.dense_unit = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.PReLU(),
            #nn.BatchNorm1d(4096),
            #nn.Linear(1024, 128),
            #nn.LeakyReLU(),
            nn.Linear(4096, 1024),
            #nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(1024,512),
            #nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(512, 256),
            #nn.Sigmoid()
        )
        '''
        self.conv1 = nn.Conv1d(input_dim, in_channels=2,out_channels=32, kernel_size=11,padding=5)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.sigmoid = nn.Sigmoid()
        '''
    def forward(self, x):
        x = self.conv_unit(x)
        x = x.view(x.size(0), -1)
        x=self.dense_unit(x)
        #x=x.squeeze()
        '''
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.max(dim=2)[0]
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        '''
        return x

# 初始化模型
input_dim = 256  # 输入维度
output_dim = 256  # 输出维度
model = MultiLabelClassifier(input_dim, output_dim)
lambda1 = lambda epoch: epoch // 30 # 第一组参数的调整方法
lambda2 = lambda epoch: 0.999 ** epoch
# 定义损失函数和优化器
#criterion = nn.BCELoss()
criterion=AsymmetricLossOptimized(disable_torch_grad_focal_loss=True)
#criterion=multilabel_categorical_crossentropy()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 80,0.5) # 选定调整方法
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True,factor=0.5, patience=10)
# 训练模型
num_epochs = 1000
model=model.cuda()
for epoch in range(num_epochs):
    model.train()
    runningloss=0.0
    lrr=optimizer.state_dict()['param_groups'][0]['lr']
    for data, target in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss=criterion(outputs,target)
        #loss = torch.sum(multilabel_categorical_crossentropy(target, outputs))
        loss.backward()
        optimizer.step()

        #runningloss+=loss.item()*target.size()[0]
        runningloss += loss.item()
    runningloss=runningloss/len(train_dataset)
    scheduler.step(runningloss)
    model.eval()

    with torch.no_grad():
        for testdata, tlabel in test_loader:
            pre=torch.nn.functional.sigmoid(model(testdata)).cpu()
            acc=evaluate_accuracy1(pre.numpy()
                                   , tlabel.cpu(), 0.68)
    print(f'lr={lrr:.6f}Epoch [{epoch + 1}/{num_epochs}], Loss: {runningloss:.6f},acc:{acc:.6f}') # 在测试集上评估

model.eval()

with torch.no_grad():
    #prediction=[]
    #correct = 0
    #total = 0
    for data, target in test_loader:
        #evaluate_accuracy()
        prediction=model(data).cpu()
    #print(f'Accuracy on test data: {100 * correct / total:.2f}%')
        np.save("predictions.npy",prediction)
        np.save("labels.npy",target.cpu())
