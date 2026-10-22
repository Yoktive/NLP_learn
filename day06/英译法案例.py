# 导包
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import device
from torch.nn import NLLLoss
from torch.utils.data import dataset, dataloader, Dataset, DataLoader
import torch.optim as optim
import time
import random
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条

# 测试当前运行时CPU还是GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'当前设备:{device}')

# todo 1.常量的定义
# 起始标记
SOS_token = 0
# 结束标记
EOS_token = 1
# 最长句子长度
MAX_LENGTH = 10
# 数据文件路径
data_path = './data/eng-fra-v2.txt'


# todo 2.字符清洗函数
def normalizeString(s):
    # 转小写，去前后空白字符
    s = s.lower().strip()
    # .!?前加空格
    s = re.sub("([.!?])", r" \1", s)
    # 过滤非字符
    s = re.sub("[^a-zA-Z.!?]+", " ", s)

    return s


# todo 3.构建文本字典
def my_getdata():
    with open(data_path, "r", encoding='utf-8') as src_f:
        lines = src_f.readlines()
        my_pairs = [[normalizeString(s) for s in line.split('\t')] for line in lines]
        # 初始化词表
        english_word2index = {'SOS': 0, 'EOS': 1}
        english_word_n = 2
        french_word2index = {'SOS': 0, 'EOS': 1}
        french_word_n = 2
        # 构建单词到索引
        for pair in my_pairs:
            for word in pair[0].split(' '):
                if word not in english_word2index:
                    english_word2index[word] = english_word_n
                    english_word_n += 1
        for pair in my_pairs:
            for word in pair[1].split(' '):
                if word not in french_word2index:
                    french_word2index[word] = french_word_n
                    french_word_n += 1
        # 构建索引到单词
        english_index2word = {v: k for k, v in english_word2index.items()}
        french_index2word = {v: k for k, v in french_word2index.items()}

        return english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs


# todo 4.构建数据集对象
english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs = my_getdata()


class MyPairsDataset(Dataset):
    # 初始化对象
    def __init__(self, my_pairs):
        self.my_pairs = my_pairs
        self.sample_len = len(my_pairs)

    # 获取数据集对象里面数据的长度
    def __len__(self):
        return self.sample_len

    # 获取单个样本方法
    def __getitem__(self, index):
        index = min(max(index, 0), self.sample_len - 1)
        # 获取句子
        x = self.my_pairs[index][0]
        y = self.my_pairs[index][1]
        # 句子文本转数值
        x = [english_word2index[word] for word in x.split(' ')]
        x.append(EOS_token)  # 句尾添加结束标记
        tensor_x = torch.tensor(x, dtype=torch.long, device=device)
        # 添加结束符号并利用GPU转为张量
        y = [french_word2index[word] for word in y.split(' ')]
        y.append(EOS_token)
        tensor_y = torch.tensor(y, dtype=torch.long, device=device)

        return tensor_x, tensor_y


# todo 5.构建数据加载器
def get_dataloader():
    # 实例化数据集对象
    my_dataset = MyPairsDataset(my_pairs)
    # 创建数据加载器
    my_dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True)
    # 测试(c测试成功后注释)
    # for i, (x, y) in enumerate(my_dataloader):
    #     print(f'x的形状:{x.shape},x的值:{x}')
    #     print(f'y的形状:{y.shape},y的值:{y}')
    #     break

    return my_dataloader


# todo 6.构建基于GRU的编码器
class EncoderRNN(nn.Module):
    # 1.参数初始化
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 实例化词嵌入层
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 实例化GRU层
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    # 2.前向传播
    def forward(self, input, hidden):
        output = self.embedding(input)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    # 3.隐藏层输入数据
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# todo 7.测试上面构建的模型是否用正常运行(已测试成功，可注释)
def dm_test_encoder():
    # 1.获取数据集对象
    my_dataloader = get_dataloader()
    # 2.初始化编码器参数
    vocab_size = english_word_n
    hidden_size = 256
    # 3.创建上面的编码器对象
    my_encoder_gru = EncoderRNN(input_size=vocab_size, hidden_size=hidden_size)
    # 4.指定用GPU来跑
    my_encoder_gru = my_encoder_gru.to(device)
    # 5.测试
    for i, (x, y) in enumerate(my_dataloader):
        print(f'x的维度{x.shape}\nx的值：{x}')
        h0 = my_encoder_gru.init_hidden()
        output, hn = my_encoder_gru(x, h0)
        print(f'output的维度:{output.shape}')
        print(f'hn的维度:{hn.shape}')
        break


# todo 8.构建基于GRU的解码器
class DecoderRNN(nn.Module):
    # 1.参数初始化
    def __init__(self, output_size, hidden_size):
        super().__init__()
        self.input_size = output_size
        self.hidden_size = hidden_size
        # 实例化词嵌入层
        self.embedding = nn.Embedding(output_size, hidden_size)
        # 实例化GRU层
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # 创建线性层
        self.out = nn.Linear(hidden_size, output_size)
        # 创建softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    # 2.前向传播
    def forward(self, input, hidden):
        output = self.embedding(input)
        # ReLU激活
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        # 线性层和softmax层处理
        output = self.softmax(self.out(output[0]))

        return output, hidden

    # 3.隐藏层输入数据
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# todo 9.测试GRU解码器(测试成功，可以注释)
def dm_test_decoder():
    # 1.获取数据加载器对象
    my_dataloader = get_dataloader()
    # 2.初始化编码器模型
    my_encoder_gru = EncoderRNN(input_size=english_word_n, hidden_size=256).to(device)
    print(f'编码器参数：{my_encoder_gru}')
    # 3.初始化解码器模型
    my_decoder_gru = DecoderRNN(output_size=french_word_n, hidden_size=256).to(device)
    print(f'解码器参数：{my_decoder_gru}')
    # 4.解码流程测试
    for i, (x, y) in enumerate(my_dataloader):
        print(f'\n输入数据英文维度：{x.shape},数据为：{x}')
        print(f'\n输入数据法语维度：{y.shape},数据为：{y}')
        # 前向传播编码
        h0 = my_encoder_gru.init_hidden()
        encoder_output_c, hidden = my_encoder_gru(x, h0)
        print(f'GRU传播后C的维度：{encoder_output_c.shape}')
        # 开始解码
        for i in range(y.shape[1]):
            # 编码后的C进行处理才可作为解码的input
            tmp = y[0][i].view(1, -1)
            output, hidden = my_decoder_gru(tmp, hidden)
            print(f'第{i + 1}个法语单词的预测结果为：{output.shape}')
        break


# todo 10.GRU带注意力机制的解码器
class AttnDecoderRNN(nn.Module):
    # todo 10.1.初始化函数
    def __init__(self, output_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        # 1.初始父类成员
        super().__init__()
        # 2.保存传入参数
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        # 3.创建词嵌入层
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # 4.注意力权重计算层
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        # 5.注意力融合层
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # 6.dropout层
        self.dropout = nn.Dropout(self.dropout_p)
        # 7.GRU层
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        # 8.输出层
        self.out = nn.Linear(self.hidden_size, self.output_size)
        # 9.对数输出层
        self.softmax = nn.LogSoftmax(dim=-1)

    # todo 10.2.前向传播
    def forward(self, input, hidden, encoder_outputs):
        # 1.词嵌入层
        embedded = self.embedding(input)
        # 2.随机失活
        embedded = self.dropout(embedded)
        # 3.计算注意力权重
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # 4.计算注意力上下文
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # 5.注意力融合层
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        # 6.激活函数处理
        output = F.relu(output)
        # 7.GRU层
        output, hidden = self.gru(output, hidden)
        # 8.输出层
        output = self.softmax(self.out(output[0]))
        # 返回结果
        return output, hidden, attn_weights

    # todo 10.3 初始化隐藏状态
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# todo 11.测试GRU带注意力机制的解码器(测试成功，可以注释)
def dm_test_attn_decoder():
    # 1.获取数据加载器对象
    my_dataloader = get_dataloader()
    # 2.初始化编码器模型
    my_encoder = EncoderRNN(input_size=english_word_n, hidden_size=256).to(device)
    # 3.初始化解码器模型
    my_decoder = AttnDecoderRNN(output_size=french_word_n, hidden_size=256).to(device)
    # 4.解码流程测试
    for i, (x, y) in enumerate(my_dataloader):
        print(f'\n输入数据英文维度：{x.shape},数据为：{x}')
        print(f'\n输入数据法语维度：{y.shape},数据为：{y}')
        hidden = my_encoder.init_hidden()
        output, hidden = my_encoder(x, hidden)
        # 储存输出
        encoder_output_c = torch.zeros(MAX_LENGTH, my_encoder.hidden_size, device=device)
        # 实际输出复制给上面的变量
        for idx in range(output.shape[1]):
            encoder_output_c[idx] = output[0, idx]
        for i in range(y.shape[1]):
            tmp = y[0][i].view(1, -1)
            output, hidden, attn_weights = my_decoder(tmp, hidden, encoder_output_c)
            print(f'解码output.shape: {output.shape}')  # [1, 4345]
            print(f'解码hidden.shape: {hidden.shape}')  # [1, 1, 256]
            print(f'解码attn_weights.shape: {attn_weights.shape}')  # [1, 10]
        break

# todo 12.构建模型内部迭代训练函数，即完成单批次训练过程
# todo 12.1 定义模型训练参数.
# 学习率，训练轮数，Teacher_Forcing比例，输出信息打印间隔(每训练1000条打印一次)，绘图间隔(每训练100条绘图一次)
my_lr, epochs, teacher_forcing_ratio, print_interval_num, plot_interval_num = 1e-4, 5, 0.5, 1000, 100
# todo 12.2 定义单批次训练函数
def train_iters(x,y,my_encoder_rnn,my_attn_decoder_rnn,myadam_encode,myadam_decode,my_crossentropy_loss):
    # 1.初始隐藏状态
    encoder_hidden=my_encoder_rnn.init_hidden()
    # 前向传播
    encoder_output,encoder_hidden=my_encoder_rnn(x,encoder_hidden)
    # 2.1 解码参数准备
    encoder_output_c=torch.zeros(MAX_LENGTH,my_encoder_rnn.hidden_size,device=device)
    # 2.2 复刻实际编码器输出
    for idx in range(x.shape[1]):
        encoder_output_c[idx]=encoder_output[0,idx]

    # 2.3 解码器初始化隐藏状态
    decoder_hidden=encoder_hidden
    # 2.4 解码器的初始输入
    input_y=torch.tensor([[SOS_token]],device=device)

    # 3.初始化损失函数
    my_loss=0.0
    y_len=y.shape[1]

    # 4.根据概率判断是否使用教师强制
    use_teacher_forcing=True if random.random()<teacher_forcing_ratio else False
    if use_teacher_forcing:
        for i in range(y_len):
            # 4.1 使用教师强制-用真实标签作为下一步的输入
            # 开始前向传播
            output_y,decoder_hidden,attn_weights = my_attn_decoder_rnn(input_y, decoder_hidden, encoder_output_c)
            # 获取当前时间步的真实标签
            target_y=y[0][i].view(1)
            # 损失累加
            my_loss+=my_crossentropy_loss(output_y,target_y)
            # 用真实标签
            input_y=y[0][i].view(1,-1)
    else:
        for i in range(y_len):
            output_y, decoder_hidden, attn_weights = my_attn_decoder_rnn(input_y, decoder_hidden, encoder_output_c)
            target_y = y[0][i].view(1)
            my_loss += my_crossentropy_loss(output_y, target_y)
            # 获取下一个预测值概率最高的
            topv,topi=output_y.topk(1)
            # 遇到结束标记停止预测
            if topi.squeeze().item()==EOS_token:
                break
            input_y=topi.detach()

    # 5.反向传播，梯度清零
    myadam_encode.zero_grad()
    myadam_decode.zero_grad()
    my_loss.backward()
    myadam_encode.step()
    myadam_decode.step()

    # 6.返回平均值
    return my_loss.item()/y_len

# todo 13.构建模型训练函数
def train_seq2seq():
    # 1.数据加载器
    my_dataloader = get_dataloader()
    # 2.编码器和解码器定义
    my_encoder_rnn = EncoderRNN(english_word_n, 256).to(device)
    my_attn_decoder_rnn = AttnDecoderRNN(french_word_n, 256, 0.1, 10).to(device)
    # 3.优化器定义
    myadam_encode=optim.Adam(my_encoder_rnn.parameters(),lr=my_lr)
    myadam_decode=optim.Adam(my_attn_decoder_rnn.parameters(),lr=my_lr)
    # 4.损失函数初始化
    my_crossentropy_loss = nn.NLLLoss()
    # 5.参数初始化,绘图专用损失值
    plot_loss_list=[]

    # 6.开始多轮训练
    for epoch_idx in range(1,epochs+1):
        print_loss_total,plot_loss_total=0.0,0.0
        start_time=time.time()
        # 内层循环
        for item,(x,y) in enumerate(tqdm(my_dataloader),start=1):
            myloss= train_iters(x,y,my_encoder_rnn,my_attn_decoder_rnn,myadam_encode,myadam_decode,my_crossentropy_loss)
            # 累加损失
            print_loss_total+=myloss
            plot_loss_total+=myloss
            # 开始打印
            if item%print_interval_num==0:
                print_loss_avg=print_loss_total/print_interval_num
                print_loss_total=0.0
                print(f'轮次：{epoch_idx},平均损失：{print_loss_avg:.4f},耗时：{time.time()-start_time:.4f}s秒')
            if item%plot_interval_num==0:
                plot_loss_avg = plot_loss_total / plot_interval_num
                plot_loss_list.append(plot_loss_avg)
                plot_loss_total = 0.0



        # 7.保存模型
        torch.save(my_encoder_rnn.state_dict(),f'./model/my_encoder_rnn_{epoch_idx}.pth')
        torch.save(my_attn_decoder_rnn.state_dict(),f'./model/my_attn_decoder_rnn_{epoch_idx}.pth')

    # 8.绘图
    plt.figure()
    plt.plot(plot_loss_list)
    plt.savefig('./img/seq2seq_loss.png')
    plt.show()

    return plot_loss_list

# todo 14.模型预测
def  evaluate_seq2seq(x,my_encoder_rnn,my_attn_dcoder_rnn):
    # 0.关闭梯度
    with torch.no_grad():
        # 1.编码结阶段
        encode_hidden=my_encoder_rnn.init_hidden()
        encode_output,encode_hidden = my_encoder_rnn(x, encode_hidden)
        # 2.解码器参数准备
        encode_output_c=torch.zeros(MAX_LENGTH,my_encoder_rnn.hidden_size,device=device)
        for idx in range(x.shape[1]):
            encode_output_c[idx]=encode_output[0,idx]
        decode_hidden=encode_hidden
        input_y=torch.tensor([[SOS_token]],device=device)
        # 3.自编码过程
        decode_words=[]
        decoder_attentions=torch.zeros(MAX_LENGTH,MAX_LENGTH)
        # 遍历
        for idx in range(MAX_LENGTH):
            output_y,decode_hidden,attn_weights = my_attn_dcoder_rnn(input_y, decode_hidden, encode_output_c)
            decoder_attentions[idx] = attn_weights
            topv,topi=output_y.topk(1)
            if topi.squeeze().item()==EOS_token:
                break
            else:
                decode_words.append(french_index2word[topi.squeeze().item()])
            input_y=topi.detach()

        return decode_words,decoder_attentions[:idx+1]

# todo 15. 模型评估函数调用，记载模型，对 自定义样本进行 翻译。
# 模型路径
PATH1 = './model/my_encoder_rnn_5.pth'
PATH2 = './model/my_attn_decoder_rnn_5.pth'

# 定义函数
def dm_test_seq2seq_evaluate():
    # 1. 获取数据加载器对象
    my_dataloader = get_dataloader()
    # 2. 加载编码器模型.
    my_encoder_rnn = EncoderRNN(english_word_n, hidden_size=256).to(device)
    # map_location: 确保在CPU和GPU(Cuda)都能加载．正常应该是用GPU训练，就用GPU预测，用CPU训练，就用CPU预测．
    # 写了 map_location能实现：用GPU训练，用CPU预测．
    # weights_only: 只加载模型权重参数．
    my_encoder_rnn.load_state_dict(torch.load(PATH1, map_location=device, weights_only=True), strict=False)
    print(f'my_encoder_rnn编码器模型架构: {my_encoder_rnn}')

    # 3. 加载解码器模型
    my_attn_decoder_rnn = AttnDecoderRNN(french_word_n, hidden_size=256).to(device)
    my_attn_decoder_rnn.load_state_dict(torch.load(PATH2, map_location=device, weights_only=True), strict=False)
    print(f'my_attn_decoder_rnn解码器模型架构: {my_attn_decoder_rnn}')

    # 4. 自定义测试样本．
    my_sample_pairs = [
        # 格式: ['英文句子', '法语句子']
        ['i m the spokesperson for this organization .', 'je suis le porte parole de cette institution .'],
        ['i am thinking about buying a new parasol .', 'je songe a acheter un nouveau parasol .'],
        ['he is able to swim very fast .', 'il est capable de nager tres vite .']
    ]
    print(f'自定义测试样本: {my_sample_pairs}')

    # 5. 对每个样本进行翻译．
    for index, pair in enumerate(my_sample_pairs):
        x = pair[0]  # 英语句子
        y = pair[1]  # 法语句子

        # 5.1 文本数值化，英语句子 → 索引序列
        tmpx = [english_word2index[word] for word in x.split()]
        tmpx.append(EOS_token)  # 添加结束标记．
        tensor_x = torch.tensor(tmpx, dtype=torch.long, device=device).view(1, -1)

        # 5.2 模型预测．
        decode_words, attentions = evaluate_seq2seq(tensor_x, my_encoder_rnn, my_attn_decoder_rnn)

        # 5.3 把翻译结果转成句子格式．
        output_sentence = ' '.join(decode_words)
        print(f'输入(原始英文句子): {x}')
        print(f'输入(原始法语句子): {y}')
        print(f'输出(翻译后的法语句子): {output_sentence}')
        print(' -.- ' * 10, '\n')

# todo 16. 绘制注意力图的函数 → Attention张量绘图(看看即可，无需编写)
def dm_test_attention():
    # 1. 获取数据加载器对象.
    my_dataloader = get_dataloader()
    # 2. 实例化模型.
    # 2.1 编码器模型
    my_encoder_rnn = EncoderRNN(english_word_n, hidden_size=256).to(device)
    my_encoder_rnn.load_state_dict(torch.load(PATH1, map_location=device, weights_only=True), strict=False)

    # 2.2 解码器模型.
    my_attn_decoder_rnn = AttnDecoderRNN(french_word_n, hidden_size=256).to(device)
    my_attn_decoder_rnn.load_state_dict(torch.load(PATH2, map_location=device, weights_only=True), strict=False)

    # 3. 定义遍历，记录：英语句子
    sentence = 'we re both teachers .'
    # 对上述的样本进行数值化.
    tmpx = [english_word2index[word] for word in sentence.split(' ')]
    tmpx.append(EOS_token)
    tensor_x = torch.tensor(tmpx, dtype=torch.long, device=device).view(1, -1)

    # 4. 模型预测.
    decode_words, attentions = evaluate_seq2seq(tensor_x, my_encoder_rnn, my_attn_decoder_rnn)
    print(f'decode_words: {decode_words}')

    # 5. 绘制注意力图.
    plt.matshow(attentions.numpy())   # 以矩阵列表的形式 显示. matrix: 矩阵.
    # 保存图像.
    plt.savefig('./img/s2s_attn.png')
    plt.show()

    # 6. 打印下参数.
    print(f'attentions.numpy(): {attentions.numpy()}')
    print(f'attentions.size()')



if __name__ == "__main__":
    # 测试todo1
    # s="  I LOve You .!? d?d 大萨达 end\n"
    # print(normalizeString(s))

    # 2. 测试数据预处理函数.
    # english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs = my_getdata()
    # print(f'英语词汇表映射: {english_word2index}\n\n\n')
    # print(f'英语单词反向映射: {english_index2word}\n\n\n')
    # print(f'英语词汇表大小: {english_word_n}\n\n\n')
    #
    # print(f'法语词汇表映射: {french_word2index}\n\n\n')
    # print(f'法语单词反向映射: {french_index2word}\n\n\n')
    # print(f'法语词汇表大小: {french_word_n}')
    # print(f'双语句子对: {my_pairs[:5]}')

    # 3. 测试数据加载器
    # get_dataloader()

    # 4. 测试GRU模型
    # dm_test_encoder()
    # dm_test_decoder()
    # dm_test_attn_decoder()

    # 5.模型训练
    # train_seq2seq()

    # 6.模型预测
    dm_test_seq2seq_evaluate()

    # 7.张量图测试
    dm_test_attention()