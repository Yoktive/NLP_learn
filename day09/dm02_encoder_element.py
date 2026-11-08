# todo 1.定义函数，进行注意力的计算
# 导包
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from dm01_input import *


from day08.dm01_input import use_position


# 定义函数计算注意力权重
def attention(query,key,value,mask=None,dropout=None):
    d_k=query.size(-1)
    scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    # 掩码是否操作
    if mask is not None:
        scores=scores.masked_fill(mask==0,-1e9)

    # 计算注意力权重
    p_attn=F.softmax(scores,dim=-1)

    # 随机失活
    if dropout is not None:
        p_attn=dropout(p_attn)

    return torch.matmul(p_attn,value),p_attn

# 测试函数
def use_attention():
    position_x=use_position()
    query=key=value=position_x

    result1,p_attn = attention(query, key, value)
    print(f'result1:{result1.shape}')
    print(f'p_attn:{p_attn.shape}')

# todo 6.定义1个克隆函数，实现指定模块克隆N次，返回N个相同模块的列表
def clones(moudle,N):
    return nn.ModuleList([copy.deepcopy(moudle) for _ in range(N)])

# todo 7.实现多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim,head,dropout_p=0.1):
        super().__init__()
        # 确保刚好可以分多个头
        assert embed_dim % head == 0
        # 计算每个头词嵌入维度
        self.d_k=embed_dim//head
        self.head=head
        # 定义4个线性层，因为多头注意力结构用到了4个线性层
        self.linears=clones(nn.Linear(embed_dim,embed_dim),4)
        # 随机失活
        self.dropout=nn.Dropout(p=dropout_p)
        # 保存注意力权重
        self.attn=None

    # 前向传播
    def forward(self,query,key,value,mask=None):
        # 是否要掩码
        if mask is not None:
            mask=mask.unsqueeze(0)
        # 获取批量大小
        self.batch=query.size(0)
        query,key,value=[
            model(x).view(self.batch,-1,self.head,self.d_k).transpose(1,2)
            for model,x in zip(self.linears,(query,key,value))
        ]
        # 多头注意力计算
        x,self.attn=attention(query, key, value, mask,self.dropout)
        # 合并
        attn_x=x.transpose(1,2).contiguous().view(self.batch,-1,self.head*self.d_k)
        # 返回结果
        return self.linears[-1](attn_x)

# todo 8.测试多头注意力
def use_multihead():
    my_attention=MultiHeadAttention(512,8)
    position_x=use_position()
    query=key=value=position_x
    mask=torch.zeros(8,4,4)
    result=my_attention(query,key,value,mask)
    print(f'(多头注意力层)：{result.shape}')

    return result

# todo 9.前馈神经网络
class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout_p=0.1):
        super().__init__()
        self.linear1=nn.Linear(d_model,d_ff)
        self.linear2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(p=dropout_p)

    # 前向传播
    def forward(self,x):
        x=self.linear1(x)
        x=self.dropout(F.relu(x))
        x=self.linear2(x)
        return x

# todo 10.测试前馈神经网络
def use_ff():
    attn_x=use_multihead()
    my_ff=FeedForward(512,2048)
    result=my_ff(attn_x)
    print(f'(前馈神经网络层)：{result.shape}')
    return result

# todo 11.规范化层
class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super().__init__()
        self.a=nn.Parameter(torch.ones(features))
        self.b=nn.Parameter(torch.zeros(features))
        self.eps=eps

    def forward(self,x):
        x_mean=x.mean(-1,keepdim=True)
        x_std=x.std(-1,keepdim=True)
        return self.a*(x-x_mean)/(x_std+self.eps)+self.b

# todo 12.测试规范化层
def use_layer_norm():
    ff_x=use_ff()
    my_layer_norm = LayerNorm(512)
    result=my_layer_norm(ff_x)
    print(f'(规范化层):{result.shape}')


if __name__=='__main__':
    # use_attention()

    use_layer_norm()