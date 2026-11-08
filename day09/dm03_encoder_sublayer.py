"""
案例:
    演示Transformer框架中的 编码器 两个子层的代码实现.

回顾: 编码器(层)是由 两个子层组成的:
    子层1:
        多头注意力子层(Multi-Head Attention) + 残差连接(Add) + 层规范化(Norm)

    子层2:
        前馈全连接子层(Feed Forward) + 残差连接(Add) + 层规范化(Norm)
"""

# 导包
from dm02_encoder_element import *
from dm01_input import *


# 1.子层连接
class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        my_result = self.dropout(self.norm(sublayer(x))) + x
        return my_result


# 2.测试子层
def use_sublayer():
    # 准备输入数据(词嵌入+位置编码)
    x = use_position()
    # 实例化子层连接结构
    sublayer_conn = SublayerConnection(d_model=512)

    # 定义子层

    # result = sublayer_conn(x, lambda x: MultiHeadAttention(512, 8)(x, x, x))
    result = sublayer_conn(x, lambda x: FeedForward(d_model=512, d_ff=2048)(x))

    print(f'子层处理结果：{result.shape}')



# todo 测试代码
if __name__=='__main__':
    use_sublayer()
