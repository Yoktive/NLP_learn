"""
    输出层的代码
"""

from dm07_decoder import *

# todo 1.定义输出部分
class Generator(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.linear=nn.Linear(d_model,vocab_size)

    def forward(self,x):
        x=self.linear(x)
        return F.log_softmax(x,dim=-1)


# todo 2.测试
def use_generator():
    result=use_decoder()
    generator=Generator(512,1000)
    output = generator(result)

    print(f'output模型最终输出结果：{output.shape},{output}')

if __name__=='__main__':
    use_generator()
