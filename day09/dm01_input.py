# 导包

import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.nn.functional import dropout


# todo 1.定义类(模拟词嵌入层)
class Embedding(nn.Module):
    # 初始化函数
    def __init__(self,vocab_size,d_model):
        super().__init__()
        self.vocab_size=vocab_size
        self.d_model=d_model
        # 单词索引映射为对应词向量
        self.embed=nn.Embedding(vocab_size,d_model)

    # 前向传播
    def forward(self,x):
        return self.embed(x)*math.sqrt(self.d_model)

# todo 2.测试Embedding词嵌入层
def dm01_embedding():
    vocab_size, d_model=1000,512
    my_embed=Embedding(vocab_size,d_model)
    x=torch.tensor([
        [100,2,421,300],
        [500,63,246,83]
    ])
    result=my_embed(x)
    print(f'结果为：{result},结果的形状：{result.shape}')

# todo 3.定义类(模拟位置编码)
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=60):
        super().__init__()
        self.dropout=nn.Dropout(p=dropout)
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len).unsqueeze(1)
        # print(f'pos的数组内容：{position}，形状：{position.shape}')
        div_term=torch.exp(torch.arange(0,d_model,2)*(-math.log(10000)/d_model))
        # print(f'div_term内容：{div_term}')
        position_value=position*div_term

        pe[:,0::2]=torch.sin(position_value)
        pe[:,1::2]=torch.cos(position_value)
        # 升维
        pe=pe.unsqueeze(0)
        # 加入缓冲区
        self.register_buffer('pe',pe)


    # 前向传播
    def forward(self,x):
        # 词向量和位置编码相加
        x=x+self.pe[:,:x.size(1)]
        return self.dropout(x)

# todo 4.测试位置编码层
def use_position():
    vocab_size,d_model=1000,512
    # 实例化Embedding层
    my_embed=Embedding(vocab_size,d_model)
    # 创建张量
    x = torch.tensor([
        [100, 2, 421, 300],
        [500, 63, 246, 83]
    ])
    # 词嵌入层先处理
    embed_x=my_embed(x)
    # 进行位置编码
    my_position=PositionalEncoding(d_model=d_model,dropout=0.1)
    position_x=my_position(embed_x)
    return position_x



# 测试主函数
if __name__=='__main__':
    # dm01_embedding()
    use_position()

