import torch
import torch.nn as nn

def fun():
    # 参1：输入维度（词向量维度），参2：输出维度（隐藏层维度），参3：隐藏层层数
    gru=nn.GRU(5,6,1)

    # 准备数据
    # 参1：句子长度，参2：批次大小（一批几个句子） ，参3：词向量维度
    input=torch.randn(1,3,5)
    # 参1：隐藏层数，参2：批次大小，参3：输出维度
    h0=torch.randn(1,3,6)
    #gru
    output,hn = gru(input, h0)

    print(f'outpu:{output},output维度：{output.shape}')      #(1,3,6)
    print(f'hn:{hn},v：{hn.shape}')                          #(1,3,6)


if __name__=='__main__':
    fun()
