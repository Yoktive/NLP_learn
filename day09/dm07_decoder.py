



# 导包
from dm06_decoder_layer import *

# todo 1.定义解码器
class Decoder(nn.Module):
    # 1.初始化函数
    def __init__(self,layer,N):
        super().__init__()
        # 克隆
        self.layers=clones(layer,N)
        # 定义规范化层
        self.norm=LayerNorm(layer.d_model)

    def forward(self,x,encoder_output,source_mask,target_mask):
        # 数据x经过多个解码器层的处理
        for layer in self.layers:
            x=layer(x,encoder_output,source_mask,target_mask)

        # 全局规范化
        return self.norm(x)

# todo 2.测试解码器
def use_decoder():
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

        # 组成解码器
        my_decoder=Decoder(my_decoder_layer,6)

        result=my_decoder(position_y,encoder_output,source_mask,target_mask)

        # 打印结果
        print(f'解码器的形状：{result.shape}')
        # print(f'解码器对象：{my_decoder}')

        return result



if __name__=='__main__':
    use_decoder()

