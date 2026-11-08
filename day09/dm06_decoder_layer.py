"""
案例:
    演示解码器层的代码实现.

解码器层由3个子层组成:
    子层1: Masked Multi-Head Attention, 掩码多头注意力机制层 + 残差连接 + 规范化层
    子层2: Multi-Head Attention, 多头注意力机制层 + 残差连接 + 规范化层
    子层3: Feed Forward, 前向传播层(前馈全连接层) + 残差连接 + 规范化层
"""
import copy

import torch

# 导包
from dm05_encoder import *

# todo 1.定义解码器层——由三个子层组成
class DecoderLayer(nn.Module):
    # 1.初始化函数
    def __init__(self,d_model,self_attn,src_attn,feed_forward,dropout=0.1):
        super().__init__()
        # 定义属性
        self.d_model=d_model
        self.self_attn=self_attn
        self.src_attn=src_attn
        self.feed_forward=feed_forward
        # 子层连接
        self.layers=clones(SublayerConnection(d_model,dropout),3)

    # 2.前向传播
    def forward(self,x,encoder_output,source_mask,target_mask):
        # 1.经过第一个子层——掩码多头注意力机制
        x=self.layers[0](x,lambda x:self.self_attn(x,x,x,target_mask))
        # 2.经过第二个子层——多头注意力机制(编码器-解码器)
        x=self.layers[1](x,lambda x:self.src_attn(x,encoder_output,encoder_output,source_mask))
        # 3.经过第三个子层——前向传播层(前馈全连接层)
        x=self.layers[2](x,self.feed_forward)
        # 4.返回结果
        return x


# todo 2.测试解码器层
def use_decoder_layer():
        y=torch.LongTensor([
            [1,2,3,4],
            [5,6,7,8]
        ])
        # 创建词嵌入层
        my_embed = Embedding(1000, 512)
        # 使用词嵌入层转词向量
        embed_y = my_embed(y)
        print(f'词嵌入层维度：{embed_y.shape}')
        # 定义位置编码
        my_position=PositionalEncoding(512,0.1)
        position_y=my_position(embed_y)
        print(f'位置编码层：{position_y}')
        # 创建多头注意力层
        multi_attn = MultiHeadAttention(512, 8)
        self_attn=copy.deepcopy(multi_attn)
        src_attn=copy.deepcopy(multi_attn)
        # 创建前向传播
        ff=FeedForward(512,2048)
        # 获取编码器输出结果
        encoder_output=use_encoder()
        # 定义掩码
        source_mask=torch.zeros(8,4,4)
        target_mask=torch.zeros(8,4,4)
        # 实例化解码器层对象
        my_decoder_layer=DecoderLayer(512,self_attn,src_attn,ff)
        result=my_decoder_layer(position_y,encoder_output,source_mask,target_mask)
        # 打印结果
        print(f'解码器层的形状：{result.shape}')
        print(f'解码器层对象：{my_decoder_layer}')



if __name__=="__main__":
    use_decoder_layer()
