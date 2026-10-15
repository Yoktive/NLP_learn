# 导包
from pydoc_data.topics import topics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import string
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

# 解决绘图时，中文乱码问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# todo 1.定义遍历，获取的常用字符数量
all_letters=string.ascii_letters+" .,;'"   #空格、逗号、点、分号、单引号
n_letters=len(all_letters)

# todo 2.获取遍历，常用的国家名种类数和个数
categories=['Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish', 'Chinese', 'Vietnamese', 'Japanese', 'French', 'Greek', 'Dutch', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German']
categories_num=len(categories)

# todo 3.获取数据函数
def read_data(file_path):
    my_list_x,my_list_y=[],[]
    with open(file_path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            if len(line)<5:
                continue
            x,y=line.strip().split('\t')
            my_list_x.append(x)
            my_list_y.append(y)
    return my_list_x,my_list_y

# todo 4.创建数据集对象，即原始数据-->数据集对象TensorDataset-->数据加载器对象DataLoader
class NameClassDataSet(Dataset):
    # 1 初始化接收到的样本和标签
    def __init__(self,my_list_x,my_list_y):
        self.my_list_x=my_list_x
        self.my_list_y=my_list_y
        self.sample_len=len(my_list_x)

    # 2 获取样本数量函数
    def __len__(self):
        return self.sample_len

    # 3 根据索引获取具体数据函数
    def __getitem__(self, index):
        # 异常处理
        index=min(max(0,index),self.sample_len-1)
        # 获取到原始数据
        x=self.my_list_x[index]     # 如x为 ding
        y=self.my_list_y[index]     # 如y为 Chinese
        # 转为one-hot
        tensor_x=torch.zeros(len(x),n_letters)
        for idx,char in enumerate(x):
            char_idx=all_letters.find(char)
            tensor_x[idx][char_idx]=1
        # 国家也转为张量
        tensor_y=torch.tensor(categories.index(y),dtype=torch.long)

        return tensor_x,tensor_y

# todo 5.获取数据加载器对象
def get_dataloader():
    # 获取数据
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')
    # 封装为对象
    name_class_dataset=NameClassDataSet(my_list_x, my_list_y)
    # 创建数据加载器对象
    my_dataloader=DataLoader(name_class_dataset,batch_size=1,shuffle=True)
    # 测试获取到的数据
    for x,y in my_dataloader:
        print(f'x.shape:{x.shape},x:{x}')
        print(f'y.shape:{y.shape},y:{y}')
        break

# todo 6.构建模型
class My_RNN(nn.Module):
    # 初始化函数
    def __init__(self,input_size,hidden_size,output_size,n_layers=1):
        super().__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.n_layers=n_layers
        # 定义RNN层
        self.rnn=nn.RNN(self.input_size,self.hidden_size,self.n_layers)
        # 定义全连接层
        self.linear=nn.Linear(self.hidden_size,self.output_size)
        # 定义激活函数
        self.softmax=nn.LogSoftmax(dim=-1)

    # 前向传播
    def forward(self,input,hidden):
        input=input.unsqueeze(1)
        # 模型开始计算
        output,hn = self.rnn(input, hidden)
        # 提取最后一个隐藏状态
        tmp_output = output[-1]
        # 通过全连接层持续输出
        tmp_output = self.linear(tmp_output)

        # 通过激活函数映射到概率分布,并返回
        return self.softmax(tmp_output),hn

    # 初始化隐藏状态
    def init_hidden(self):
        return torch.zeros(self.n_layers,1,self.hidden_size)

# 构建LSTM模型
class My_LSTM(nn.Module):
    # 初始化函数
    def __init__(self,input_size,hidden_size,output_size,n_layers=1):
        super().__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.n_layers=n_layers
        # 定义RNN层
        self.rnn=nn.LSTM(self.input_size,self.hidden_size,self.n_layers)
        # 定义全连接层
        self.linear=nn.Linear(self.hidden_size,self.output_size)
        # 定义激活函数
        self.softmax=nn.LogSoftmax(dim=-1)

    # 前向传播
    def forward(self,input,hidden,c):
        input=input.unsqueeze(1)
        # 模型开始计算
        output,(hn,cn) = self.rnn(input, (hidden,c))
        # 提取最后一个隐藏状态
        tmp_output = output[-1]
        # 通过全连接层持续输出
        tmp_output = self.linear(tmp_output)

        # 通过激活函数映射到概率分布,并返回
        return self.softmax(tmp_output),hn,cn

    # 初始化隐藏状态
    def init_hidden(self):
        hidden=c=torch.zeros(self.n_layers, 1, self.hidden_size)
        return hidden,c

# 构建GRU模型
class My_GRU(nn.Module):
    # 初始化函数
    def __init__(self,input_size,hidden_size,output_size,n_layers=1):
        super().__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.n_layers=n_layers
        # 定义RNN层
        self.rnn=nn.GRU(self.input_size,self.hidden_size,self.n_layers)
        # 定义全连接层
        self.linear=nn.Linear(self.hidden_size,self.output_size)
        # 定义激活函数
        self.softmax=nn.LogSoftmax(dim=-1)

    # 前向传播
    def forward(self,input,hidden):
        input=input.unsqueeze(1)
        # 模型开始计算
        output,hn = self.rnn(input, hidden)
        # 提取最后一个隐藏状态
        tmp_output = output[-1]
        # 通过全连接层持续输出
        tmp_output = self.linear(tmp_output)

        # 通过激活函数映射到概率分布,并返回
        return self.softmax(tmp_output),hn

    # 初始化隐藏状态
    def init_hidden(self):
        return torch.zeros(self.n_layers,1,self.hidden_size)

# todo 7.模型可用性测试
def dm_test_myrnn():
    my_rnn = My_RNN(57, 128, 18)
    # 准备数据
    input=torch.randn(6,57)
    h0=my_rnn.init_hidden()
    output,hn=my_rnn(input,h0)
    print(f'输出数据的形状：{output.shape}\n输出的数据本身：{output}')
    print(f'隐藏层的形状：{hn.shape}\n输出的隐藏层本身：{hn}')

# todo 8.模型带数据测试RNN、LSTM、GRU
def dm_test_rnn_lstm_gru():
    # 1：定义变量记录输入层、隐藏层、输出层维度
    input_size,n_hidden,output_size=n_letters,128,categories_num
    # 2：加载数据
    my_list_x,my_list_y = read_data('./data/name_classfication.txt')
    # 3：创建数据集对象
    name_class_dataset = NameClassDataSet(my_list_x, my_list_y)
    # 4:创建数据加载器
    my_dataloader = DataLoader(name_class_dataset, batch_size=1, shuffle=True)
    # 5：模型初始化
    my_rnn = My_RNN(input_size, n_hidden, output_size)
    my_lstm = My_LSTM(input_size, n_hidden, output_size)
    my_gru = My_GRU(input_size, n_hidden, output_size)
    # 6：模型结构可视化
    # print(f'RNN的模型结构:{my_rnn}')
    # print(f'LSTM的模型结构:{my_lstm}')
    # print(f'GRU的模型结构:{my_gru}')
    # 7.1：RNN模型测试
    for i ,(x,y) in enumerate(my_dataloader):
        # print(f'第几条编号：{i}')
        # print(f'输入数据x：{x}，输入数据的维度：{x.shape}')
        # print(f'输出数据y：{y}，输出数据的维度：{y.shape}')
        # 初始化隐藏状态
        hidden=my_rnn.init_hidden()
        # 前向传播
        output,hidden=my_rnn(x[0],hidden)
        print(f'RNN输出形状：{output.shape},预测结果:{output}')
        if i==1:
            break

    # 7.2LSTM模型测试
    for i, (x, y) in enumerate(my_dataloader):
        # 初始化隐藏状态
        hidden,c= my_lstm.init_hidden()
        # 前向传播
        output, hidden,c = my_lstm(x[0], hidden,c)
        print(f'LSTM输出形状：{output.shape},预测结果:{output}')
        if i == 0:
            break

# todo 9.模型训练
my_lr,epochs=1e-3,1

# todo 9.1 RNN模型训练
def train_rnn():

    # 1.准备数据
    # 1.1 读取数据
    my_list_x,my_list_y=read_data('./data/name_classfication.txt')
    # 1.2 构建数据集对象
    name_class_dataset = NameClassDataSet(my_list_x, my_list_y)

    # 2.模型与优化器初始化
    # 2.1 定义模型参数
    input_size,n_hidden,output_size=n_letters,128,categories_num
    # 2.2 创建模型对象
    my_rnn = My_RNN(input_size, n_hidden, output_size)
    # 2.3 定义损失函数和优化器
    criterion=nn.NLLLoss()
    optimizer=optim.Adam(my_rnn.parameters(),lr=my_lr)

    # 3.训练开始前参数初始化
    start_time=time.time()
    total_iter_num=0               # 已训练的样本数
    total_loss=0.0                 # 已训练的损失和
    total_loss_list=[]             # 每100条样本的平均损失
    total_acc_num=0                # 预测准确的样本数
    total_acc_list=[]              # 每100条样本的平均准确率

    # 4.具体训练过程
    for epoch in range(epochs):
        print(f'\n开始RNN模型第{epoch+1}/{epochs}轮的训练')
        # 4.1 创建数据集加载器对象
        train_dataloader = DataLoader(name_class_dataset, batch_size=1, shuffle=True)
        # 4.2 样本开始迭代训练
        for i,(x,y) in enumerate(tqdm(train_dataloader)):
            # 4.3 前向传播，计算结果
            output,hidden = my_rnn(x[0], my_rnn.init_hidden())
            # 4.4 计算损失
            my_loss = criterion(output, y)
            # 4.5 三剑客:梯度清零+反向传播+优化器更新
            optimizer.zero_grad()
            my_loss.backward()
            optimizer.step()
            # 4.6 统计训练结果
            total_iter_num+=1
            total_loss+=my_loss.item()
            # 4.7 计算当前样本的计算准确率
            pre_tag = torch.argmax(output).item()
            total_acc_num+=(1 if pre_tag==y else 0)
            # 4.8 统计：每100个样本计算一次平均损失，准确率
            if total_iter_num%100==0:
                avg_loss=total_loss/total_iter_num
                total_loss_list.append(avg_loss)
                avg_acc=total_acc_num/total_iter_num
                total_acc_list.append(avg_acc)
            # 4.9 统计：每2000个样本，打印训练日志
            if total_iter_num%2000==0:
                avg_loss=total_loss/total_iter_num
                end_time=int(time.time()-start_time)
                print(f'轮次：{epoch},训练样本数：{total_iter_num},平均损失：{avg_loss},耗时：{end_time}s，准确率：{total_acc_num/total_iter_num}')

        # 4.10 保存模型
        torch.save(my_rnn.state_dict(),f'./model/my_rnn_model_{epoch+1}.bin')

    # 5.返回统计结果
    total_time=int(time.time()-start_time)
    print(f'训练完成，总耗时：{total_time}s,总训练了{total_iter_num}个样本')

    # 6 返回结果
    return total_loss_list,total_time,total_acc_list

# todo 9.2 LSTM模型训练
def train_lstm():

    # 1.准备数据
    # 1.1 读取数据
    my_list_x,my_list_y=read_data('./data/name_classfication.txt')
    # 1.2 构建数据集对象
    name_class_dataset = NameClassDataSet(my_list_x, my_list_y)

    # 2.模型与优化器初始化
    # 2.1 定义模型参数
    input_size,n_hidden,output_size=n_letters,128,categories_num
    # 2.2 创建模型对象
    my_rnn = My_LSTM(input_size, n_hidden, output_size)
    # 2.3 定义损失函数和优化器
    criterion=nn.NLLLoss()
    optimizer=optim.Adam(my_rnn.parameters(),lr=my_lr)

    # 3.训练开始前参数初始化
    start_time=time.time()
    total_iter_num=0               # 已训练的样本数
    total_loss=0.0                 # 已训练的损失和
    total_loss_list=[]             # 每100条样本的平均损失
    total_acc_num=0                # 预测准确的样本数
    total_acc_list=[]              # 每100条样本的平均准确率

    # 4.具体训练过程
    for epoch in range(epochs):
        print(f'\n开始LSTM的第{epoch+1}/{epochs}轮的训练')
        # 4.1 创建数据集加载器对象
        train_dataloader = DataLoader(name_class_dataset, batch_size=1, shuffle=True)
        # 4.2 样本开始迭代训练
        for i,(x,y) in enumerate(tqdm(train_dataloader)):
            # 4.3 前向传播，计算结果
            hidden,c=my_rnn.init_hidden()
            output,hidden,c= my_rnn(x[0], hidden,c)
            # 4.4 计算损失
            my_loss = criterion(output, y)
            # 4.5 三剑客:梯度清零+反向传播+优化器更新
            optimizer.zero_grad()
            my_loss.backward()
            optimizer.step()
            # 4.6 统计训练结果
            total_iter_num+=1
            total_loss+=my_loss.item()
            # 4.7 计算当前样本的计算准确率
            pre_tag = torch.argmax(output).item()
            total_acc_num+=(1 if pre_tag==y else 0)
            # 4.8 统计：每100个样本计算一次平均损失，准确率
            if total_iter_num%100==0:
                avg_loss=total_loss/total_iter_num
                total_loss_list.append(avg_loss)
                avg_acc=total_acc_num/total_iter_num
                total_acc_list.append(avg_acc)
            # 4.9 统计：每2000个样本，打印训练日志
            if total_iter_num%2000==0:
                avg_loss=total_loss/total_iter_num
                end_time=int(time.time()-start_time)
                print(f'轮次：{epoch},训练样本数：{total_iter_num},平均损失：{avg_loss},耗时：{end_time}s，准确率：{total_acc_num/total_iter_num}')

        # 4.10 保存模型
        torch.save(my_rnn.state_dict(),f'./model/my_lstm_model_{epoch+1}.bin')

    # 5.返回统计结果
    total_time=int(time.time()-start_time)
    print(f'训练完成，总耗时：{total_time}s,总训练了{total_iter_num}个样本')

    # 6 返回结果
    return total_loss_list,total_time,total_acc_list

# todo 9.3 GRU模型训练
def train_gru():

    # 1.准备数据
    # 1.1 读取数据
    my_list_x,my_list_y=read_data('./data/name_classfication.txt')
    # 1.2 构建数据集对象
    name_class_dataset = NameClassDataSet(my_list_x, my_list_y)

    # 2.模型与优化器初始化
    # 2.1 定义模型参数
    input_size,n_hidden,output_size=n_letters,128,categories_num
    # 2.2 创建模型对象
    my_rnn = My_GRU(input_size, n_hidden, output_size)
    # 2.3 定义损失函数和优化器
    criterion=nn.NLLLoss()
    optimizer=optim.Adam(my_rnn.parameters(),lr=my_lr)

    # 3.训练开始前参数初始化
    start_time=time.time()
    total_iter_num=0               # 已训练的样本数
    total_loss=0.0                 # 已训练的损失和
    total_loss_list=[]             # 每100条样本的平均损失
    total_acc_num=0                # 预测准确的样本数
    total_acc_list=[]              # 每100条样本的平均准确率

    # 4.具体训练过程
    for epoch in range(epochs):
        print(f'\n开始GRU模型第{epoch+1}/{epochs}轮的训练')
        # 4.1 创建数据集加载器对象
        train_dataloader = DataLoader(name_class_dataset, batch_size=1, shuffle=True)
        # 4.2 样本开始迭代训练
        for i,(x,y) in enumerate(tqdm(train_dataloader)):
            # 4.3 前向传播，计算结果
            output,hidden = my_rnn(x[0], my_rnn.init_hidden())
            # 4.4 计算损失
            my_loss = criterion(output, y)
            # 4.5 三剑客:梯度清零+反向传播+优化器更新
            optimizer.zero_grad()
            my_loss.backward()
            optimizer.step()
            # 4.6 统计训练结果
            total_iter_num+=1
            total_loss+=my_loss.item()
            # 4.7 计算当前样本的计算准确率
            pre_tag = torch.argmax(output).item()
            total_acc_num+=(1 if pre_tag==y else 0)
            # 4.8 统计：每100个样本计算一次平均损失，准确率
            if total_iter_num%100==0:
                avg_loss=total_loss/total_iter_num
                total_loss_list.append(avg_loss)
                avg_acc=total_acc_num/total_iter_num
                total_acc_list.append(avg_acc)
            # 4.9 统计：每2000个样本，打印训练日志
            if total_iter_num%2000==0:
                avg_loss=total_loss/total_iter_num
                end_time=int(time.time()-start_time)
                print(f'轮次：{epoch},训练样本数：{total_iter_num},平均损失：{avg_loss},耗时：{end_time}s，准确率：{total_acc_num/total_iter_num}')

        # 4.10 保存模型
        torch.save(my_rnn.state_dict(),f'./model/my_gru_model_{epoch+1}.bin')

    # 5.返回统计结果
    total_time=int(time.time()-start_time)
    print(f'训练完成，总耗时：{total_time}s,总训练了{total_iter_num}个样本')

    # 6 返回结果
    return total_loss_list,total_time,total_acc_list

# todo 10 绘图
def dm_test_train_rnn_lstm_gru():
    # 1. 训练3种模型，并获取性能指标.
    # 参1: 损失列表，参2: 训练总耗时，参3: 准确率列表.
    total_loss_list_rnn, total_time_rnn, total_acc_list_rnn = train_rnn()
    total_loss_list_lstm, total_time_lstm, total_acc_list_lstm = train_lstm()
    total_loss_list_gru, total_time_gru, total_acc_list_gru = train_gru()

    # 2. 绘制 损失对比曲线(评估: 模型收敛速度)
    # 2.1 创建画布.
    plt.figure(num=0, figsize=(10, 5))
    # 2.2 绘制各模型损失曲线.
    plt.plot(total_loss_list_rnn, label='RNN')
    plt.plot(total_loss_list_lstm, label='LSTM')
    plt.plot(total_loss_list_gru, label='GRU')
    # 2.3 设置图表属性
    plt.title('模型损失对比曲线')
    plt.xlabel('训练步数(每100步)')
    plt.ylabel('平均损失值')
    plt.grid(visible=True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.savefig('./img/RNN_LSTM_GRU_loss_time.png')
    plt.show()

    # 3. 绘制 训练耗时对比柱状图(评估: 模型计算效率)
    # 3.1 创建画布, 1: 图2
    plt.figure(num=1, figsize=(10, 5))
    # 3.2 准备x轴 和 y轴标签内容.
    x_data = ['RNN', 'LSTM', 'GRU']
    y_data = [total_time_rnn, total_time_lstm, total_time_gru]
    # 3.3 绘制柱状图
    plt.bar(range(len(x_data)), y_data, tick_label=x_data)
    # 3.4 设置图表属性
    plt.title('模型耗时对比柱状图')
    plt.savefig('./img/RNN_LSTM_GRU_time.png')
    plt.show()

    # 4. 绘制 训练准确率对比曲线(评估: 模型效果)
    # 4.1 创建画布. 2: 图3
    plt.figure(num=2, figsize=(10, 5))
    # 4.2 绘制各模型准确率曲线.
    plt.plot(total_acc_list_rnn, label='RNN', color='red')
    plt.plot(total_acc_list_lstm, label='LSTM', color='green')
    plt.plot(total_acc_list_gru, label='GRU', color='orange')
    # 4.3 绘制图表属性.
    plt.title('模型准确率对比曲线')
    plt.legend(loc='upper left')
    plt.savefig('./img/RNN_LSTM_GRU_acc.png')
    plt.show()

# todo 11 RNN模型预测

# 1.模型参数路径
my_rnn_path='./model/my_rnn_model_1.bin'
my_lstm_path='./model/my_lstm_model_1.bin'
my_gru_path='./model/my_gru_model_1.bin'
# 2.定义函数将字符转为one-hot编码
def lineToTensor(line):
    tensor_x=torch.zeros(len(line),n_letters)
    for idx,char in enumerate(line):
        char_idx=all_letters.find(char)
        tensor_x[idx][char_idx]=1
    return tensor_x

def dm_predict_rnn(x):
    n_letters,n_hidden,n_categories=57,128,18
    x_tensor=lineToTensor(x)
    my_rnn=My_RNN(n_letters,n_hidden,n_categories)
    my_rnn.load_state_dict(torch.load(my_rnn_path))
    with torch.no_grad():
        output,hidden=my_rnn(x_tensor,my_rnn.init_hidden())
        topv,topi=output.topk(3,1,True)
        print(f'待预测的文本;{x}')
        for i in range(3):
            value=topv[0][i].item()
            category_idx=topi[0][i].item()
            category=categories[category_idx]
            print(f'value：{value}|类别：{category}')

if __name__=='__main__':
    # train_rnn()
    # train_lstm()
    # train_gru()
    dm_predict_rnn('xie')