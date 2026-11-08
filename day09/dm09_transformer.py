"""
案例:
    演示Transformer整体模型架构.
"""
import copy

# 导包
from dm08_output import *


# todo 1. 定义整体的Transformer模型架构, 初始化: Transformer模型的主要组件.
class EncoderDecoder(nn.Module):
    # 1. 初始化函数.
    def __init__(self, source_embed, encoder, target_embed, decoder, generator):
        """
        初始化Transformer框架的基本参数
        :param source_embed: 编码器的输入嵌入层, 包含: 词向量 和 位置编码
        :param encoder: 编码器模块
        :param target_embed: 解码器的输入嵌入层, 包含: 词向量 和 位置编码
        :param decoder: 解码器模块
        :param generator: 输出层,  将解码器的输出 转换为 词汇表的概率分布
        """
        # 1. 初始化父类成员.
        super().__init__()

        # 2. 定义参数, 记录上述的变量.
        self.source_embed = source_embed
        self.encoder = encoder
        self.target_embed = target_embed
        self.decoder = decoder
        self.generator = generator


    # 2. 前向传播
    def forward(self, source_x, target_y, source_mask, target_mask):
        """
        (编码解码器)前向传播函数, Transformer模型的前向传播过程, 即: 先编码, 后解码.
        :param source_x: 编码器的输入, 例如: [batch_size, seq_len] -> [2, 4]
        :param target_y: 解码器的输入, 例如: [batch_size, seq_len] -> [2, 4]
        :param source_mask: padding mask机制, 防止填充的pad值影响注意力计算, 例如: [8, 4, 4]
        :param target_mask: sentence mask机制, 防止未来的信息被提前利用.
        :return: 模型预测的词汇表的概率分布, 即: [2, 4, 1000]
        """
        # 1. 先得到编码器的输出结果.
        encoder_result = self.encode(source_x, source_mask)

        # 2. 再得到解码器的输出结果.
        decoder_result = self.decode(target_y, encoder_result, source_mask, target_mask)

        # 3. 将解码器的结果 进行输出层处理, 并输出.
        output = self.generator(decoder_result)

        # 4. 返回结果.
        return output


    # 3. 得到编码器的输出结果, 即:编码器的前向传播过程.
    def encode(self, source_x, source_mask):
        """
        得到编码器的输出结果, 即: 编码器前向传播过程.
        :param source_x: 编码器的输入, [batch_size, seq_len] -> [2, 4]
        :param source_mask: 输入序列的掩码, [n_head, seq_len, seq_len] -> [2, 4, 4], padding-mask: 防止填充的pad值影响注意力计算结果.
        :return: encoder_output 编码器的输出 -> [batch_size, seq_len, d_model] -> [2, 4, 512]
        """
        # 1. 对source_x进行词嵌入处理, 转成向量.
        embed_x = self.source_embed(source_x)
        # 2. 通过编码器处理嵌入后的数据.
        return self.encoder(embed_x, source_mask)


    # 4. 获取解码器的输出结果, 即:解码器的前向传播过程.
    def decode(self, target_y, encoder_output, source_mask, target_mask):
        """
        得到解码器的输出结果, 即: 解码器的前向传播过程.
        :param target_y: 解码器的输入, 例如: [batch_size, seq_len] -> [2, 4]
        :param encoder_output: 编码器的输出结果, [batch_size, seq_len, d_model] -> [2, 4, 512]
        :param source_mask: 代表padding mask, 防止填充的pad值影响注意力计算结果.
        :param target_mask: 代表sentence mask, 防止未来的信息被提前利用(防止偷看未来的词)
        :return: 解码器的输出结果, [batch_size, seq_len, d_model] -> [2, 4, 512]
        """
        # 1. 对target_y进行词嵌入处理, 转成向量.
        embed_y = self.target_embed(target_y)
        # 2. 通过解码器处理嵌入后的数据.
        return self.decoder(embed_y, encoder_output, source_mask, target_mask)



# todo 2. 测试Transformer模型架构.
def make_model():
    # 1. 定义深拷贝工具函数, 用于复制模块.
    c = copy.deepcopy

    # ------------------------------------ 编码器层 ------------------------------------
    # 2. 编码器, 词嵌入层.
    source_embed = Embedding(vocab_size=1000, d_model=512)
    # 3. 编码器, 位置编码层.
    source_position = PositionalEncoding(d_model=512, dropout=0.1)
    # 4. 多头注意力机制 -> 实例化.
    self_attn = MultiHeadAttention(embed_dim=512, head=8)
    # 5. 定义前馈全连接层.
    ff = FeedForward(d_model=512, d_ff=2048)
    # 6. 设置随机失活概率, 防止: 过拟合.
    dropout_p = 0.2
    # 7. 单个编码器层的实例化, 包括: 多头注意力机制, 前馈全连接层, 残差连接, 规范化层
    my_encoder_layer = EncoderLayer(d_model=512, self_attn=self_attn, feed_forward=ff, dropout=dropout_p)
    # 8. 编码器的初始化.
    my_encoder = Encoder(my_encoder_layer, N=6)


    # ------------------------------------ 解码器层 ------------------------------------
    # 1. 解码器, 词嵌入层.
    # target_embed = Embedding(vocab_size=1000, d_model=512)      # 思路1: 手动写一遍.
    # target_embed = copy.deepcopy(source_embed)                  # 思路2: 深拷贝.
    target_embed = c(source_embed)                                # 思路3: 深拷贝 -> 语法糖...

    # 2. 解码器, 位置编码层.
    target_position = c(source_position)

    # 3. 解码器的自注意力机制, 处理: 目标序列内部关系.
    self_attn1 = c(self_attn)
    # 4. 解码器-编码器注意力机制: 处理: 目标序列与源序列之间的关系.
    source_attn1 = c(self_attn)
    # 5. 定义前馈全连接层.
    ff1 = c(ff)
    # 6. 单个解码器层的实例化, 包括: 多头注意力机制, 前馈全连接层, 残差连接, 规范化层
    my_decoder_layer = DecoderLayer(d_model=512, self_attn=self_attn1, src_attn=source_attn1, feed_forward=ff1, dropout=dropout_p)
    # 7. 解码器的初始化.
    my_decoder = Decoder(my_decoder_layer, N=6)


    # ------------------------------------ 输出层 ------------------------------------
    # 1. 输出层, 实例化.
    generator = Generator(d_model=512, vocab_size=1000)



    # ------------------------------------ 组装完整的Transformer模型 -------------------
    # 编码器输入处理: 词嵌入 + 位置编码
    # 解码器输入处理: 词嵌入 + 位置编码
    my_transformer = EncoderDecoder(
        nn.Sequential(source_embed, source_position),       # 编码器输入处理链
        my_encoder,         # 编码器
        nn.Sequential(target_embed, target_position),       # 解码器输入处理链
        my_decoder,         # 解码器
        generator           # 输出层
    )

    # 打印模型结构
    print(my_transformer)


    # ------------------------------------ Transformer模型的 前向测试 -------------------
    # 1. 构建测试输入数据 -> 源序列, 假设: 2个句子, 每个句子有4个单词(token)
    source_x = torch.LongTensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    # 2. 构建目标输入数据 -> 目标序列, 假设: 2个句子, 每个句子有4个单词(token)
    target_y = torch.LongTensor([[3, 8, 6, 4], [9, 6, 2, 6]])

    # 3. 初始化掩码(实际开发要根据序列长度 动态生成)
    source_mask = torch.zeros(8, 4, 4)
    target_mask = c(source_mask)

    # 4. 执行前向传播.
    print(' -.- ' * 10)
    result = my_transformer(source_x, target_y, source_mask, target_mask)
    print(f'result是Transformer的输出结果: {result}')
    print(f'result的形状: {result.shape}')     # [2, 4, 1000]



# todo 3. 测试代码
if __name__ == '__main__':
    make_model()