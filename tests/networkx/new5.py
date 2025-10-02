"""
构建网络工具包
迁移学习工具包
author：zhaoxuefeng
datetime：2021-09-20 16:30:57
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Sequential的具体实现
class MySequential(nn.Module):
    def __init__(self,*args):
        """
        Sequential的具体实现
        :param args:
        """
        super(MySequential, self).__init__()
        for block in args:
            self._modules[block] = block

    def forward(self,X):
        for block in self._modules.values():
            X = block(X)
        return X





# The sample#########

class Reshape(nn.Module):
    def forward(self,X):
        return X.view(-1,1,28,28)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.one = nn.Linear(20,256)

    def forward(self,X):
        return self.one(X)

net = nn.Sequential(nn.Linear(12,20),nn.ReLU(),nn.Linear(20,1))

#当然，网络不仅仅可以实现对于既定的层的使用，例如Linear
#还可以有对于一些自定义矩阵的支持

class FixedHiddenMlp(nn.Module):
    def __init__(self):
        super(FixedHiddenMlp, self).__init__()
        self.rand_weight = torch.randn((20,20),requires_grad=False)
        #这里随机生成W，至于后面的传参还不知道
        self.linear = nn.Linear(20,20)
    def forward(self,X):
        X = self.linear(X)
        X = F.relu(torch.mm(X,self.rand_weight)+1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /=2
        return X.sum()

#也可以自定义层

#不带参数的层¶
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))


# 带参数的层

class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


linear = MyLinear(5, 3)
linear.weight





# 参数管理
#1当通过Sequential类定义模型时，我们可以利用索引访问模型的任意层

def param_show(net):
    print(net[0].state_dict())
    print(net[0].weight)
    print(net[0].bias)
    print(net[0].bias.data)
    print(net[0].weight.grad)

#一次性访问所有参数
def param_show_all(net):
    print(*[(name, param.shape) for name, param in net[0].named_parameters()])
    print(*[(name, param.shape) for name, param in net.named_parameters()])


# 参数初始化

#1定义初始化函数：
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)

        # nn.init.constant_(m.weight, 1) # 初始化为常数
        # nn.init.xavier_uniform_(m.weight)
        # nn.init.constant_(m.weight, 42)
        nn.init.zeros_(m.bias)

def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


net.apply(init_normal)


#修改参数的值
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]

# 参数绑定，在多个层之间共享参数


# 我们需要给共享层一个名称，以便可以引用它的参数。
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))

# shared 就是共享层
"""
这个例子表明第二层和第三层的参数是绑定的。
它们不仅值相等，而且由相同的张量表示。
你可能会想，当参数绑定时，梯度会发生什么情况？
答案是由于模型参数包含梯度，
因此在反向传播期间第二个隐藏层和第三个隐藏层的梯度会加在一起。
"""


#冻结参数 , 将需要冻结的层设置为requires_grad = False

# 然后需要在优化器中filter一下
params = filter(lambda p: p.requires_grad, net.parameters())
optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.0005)


# 参数的保存




##迁移学习

pretrained_net = torchvision.models.resnet18(pretrained=True)

print(pretrained_net)#self里的都包含

print(pretrained_net.fc)

pretrained_net = nn.Linear(pretrained_net.fc.in_features,2)

nn.init.xavier_uniform_(pretrained_net.fc.weight)

params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
trainer = torch.optim.SGD([{'params': params_1x},{'params': net.fc.parameters(),'lr': 0.001}],
                          lr=0.0001,
                          weight_decay=0.001)



# nn.init.xavier_uniform_(m.weight)
#
# net.apply(init_weights)  # 初始化参数
#
# print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
#       f'test acc {test_acc:.3f}')
