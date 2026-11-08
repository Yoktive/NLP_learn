""" 编码器层的搭建 """
import torch

from dm01_input import *
from dm02_encoder_element import *
from dm03_encoder_sublayer import *


# todo 1.初始化编码器层
class EncoderLayer(nn.Module):
    # 1.初始化函数
    def __init__(self, d_model, self_attn, feed_forward, dropout=0.1):
        super().__init__()
        # 保存参数
        self.d_model = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 克隆自连接结构
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    # 前向传播
    def forward(self, x, mask):
        # 自注意力层
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 子连接层
        x = self.sublayer[1](x, lambda x: self.feed_forward(x))
        # 返回结果
        return x


# todo 2.测试编码器层
def use_encoder_layer():
    # 1.准备数据
    x = use_position()

    # 2.实例化子层对象
    # 2.1 多头注意力对象
    multi_head = MultiHeadAttention(512, 8)

    # 2.2 前馈全连接对象
    feed_forward = FeedForward(512, 2048)

    # 3.创建编码器对象
    encoder_layer = EncoderLayer(512, multi_head, feed_forward)
    # 4.构建掩码张量
    mask = torch.zeros(8, 4, 4)
    # 5.前向传播
    output = encoder_layer(x, mask)
    # 6.打印结果
    print(f'编码器输出形状：{output.shape}')
    print('===' * 40)
    print(f'编码器对象：{encoder_layer}')


if __name__ == '__main__':
    use_encoder_layer()
