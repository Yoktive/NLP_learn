"""
案例:
    代码演示 编码器的代码实现.

总结:
    1. 编码器 默认有6个 编码器层.
    2. 每个 编码器层 默认有2个子层.
        多头注意力子层
        前向传播子层
    3. 还有 残差连接 + 规范化层
"""
import torch

# 导包
from dm04_encoder_layer import *


# todo 1.编码器层
class Encoder(nn.Module):
    # 1.初始化函数
    def __init__(self, layer, N):
        super().__init__()
        # 2.克隆N个编码器层对象
        self.layers = clones(layer, N)
        # 3.定义最终规范化层
        self.norm = LayerNorm(layer.d_model)

    # 2.定义前向传播
    def forward(self, x, mask):
        # 1.x结果N个编码器层处理
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# todo 2.测试编码器层
def use_encoder():
    # 1.获取数据
    x = use_position()

    # 2.创建单个编码器层
    # 2.1 创建多头注意力
    multi_head = MultiHeadAttention(512, 8)
    # 2.2 创建前馈全连接层
    feed_forward = FeedForward(512, 2048)
    # 2.3 创建编码器层对象
    encoder_layer = EncoderLayer(512, multi_head, feed_forward)

    # 3 实例化编码器
    encoder = Encoder(encoder_layer, 3)

    # 4.创建掩码
    mask = torch.zeros(8, 4, 4)

    # 5.执行编码过程
    encoder_output = encoder(x, mask)

    # 6.打印结果
    print(f'编码器最终输出形状：{encoder_output.shape}')
    print(f'x输入前部分数据：{x[0, 0, :5]}')
    print(f'x输出后部分数据：{encoder_output[0, 0, :5]}')

    return encoder_output


if __name__=='__main__':
    use_encoder()
