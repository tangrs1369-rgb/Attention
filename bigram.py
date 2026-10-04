import torch # 导入 PyTorch 主库
import torch.nn as nn # 导入神经网络模块，简写为 nn
from torch.nn import functional as F # 导入函数式接口（softmax、cross_entropy 等），简写为 F

# 超参数
batch_size = 32 # 每个批次并行处理多少条独立的序列
block_size = 8 # 预测时最多能看到的上下文长度（序列长度）
max_iters = 3000 # 训练总步数
eval_interval = 300 # 每隔多少步评估一次 loss
learning_rate = 1e-2 # 学习率（bigram 模型很小，可以用较大的学习率）
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 有 GPU 就用 GPU，否则用 CPU
eval_iters = 200 # 评估 loss 时取多少个批次求平均
# ------------

torch.manual_seed(1337) # 固定随机种子，保证结果可复现

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt # 数据集下载地址
with open('input.txt', 'r', encoding='utf-8') as f: # 以 UTF-8 编码打开数据集文件
    text = f.read() # 把整个文件读成一个字符串

# 找出文本中出现过的所有不重复字符
chars = sorted(list(set(text))) # 去重后排序，得到字符表
vocab_size = len(chars) # 词表大小（字符种类数，Tiny Shakespeare 为 65）
# 建立字符和整数之间的映射
stoi = { ch:i for i,ch in enumerate(chars) } # 字符 -> 整数 的字典
itos = { i:ch for i,ch in enumerate(chars) } # 整数 -> 字符 的字典
encode = lambda s: [stoi[c] for c in s] # 编码器：输入字符串，输出整数列表
decode = lambda l: ''.join([itos[i] for i in l]) # 解码器：输入整数列表，输出字符串

# 划分训练集和验证集
data = torch.tensor(encode(text), dtype=torch.long) # 把整个文本编码成一维整数张量
n = int(0.9*len(data)) # 前 90% 作为训练集，剩下的作为验证集
train_data = data[:n] # 训练集
val_data = data[n:] # 验证集

# 数据加载
def get_batch(split): # 根据 split（'train' 或 'val'）取一个批次的数据
    # 生成一小批输入 x 和目标 y
    data = train_data if split == 'train' else val_data # 选择训练集或验证集
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 随机选 batch_size 个起始位置
    x = torch.stack([data[i:i+block_size] for i in ix]) # 输入：从每个起始位置截取 block_size 个字符，堆叠成 (B,T)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # 目标：向右错开一位，即每个位置的“下一个字符”
    x, y = x.to(device), y.to(device) # 把数据搬到 GPU/CPU 上
    return x, y # 返回输入和目标

@torch.no_grad() # 装饰器：该函数内不计算梯度，节省内存和计算
def estimate_loss(): # 在训练集和验证集上估计平均 loss
    out = {} # 存放结果的字典
    model.eval() # 切换到评估模式
    for split in ['train', 'val']: # 分别评估训练集和验证集
        losses = torch.zeros(eval_iters) # 用来记录每个批次的 loss
        for k in range(eval_iters): # 取 eval_iters 个批次
            X, Y = get_batch(split) # 取一个批次
            logits, loss = model(X, Y) # 前向传播得到 loss
            losses[k] = loss.item() # 记录该批次的 loss（转成 Python 数值）
        out[split] = losses.mean() # 求平均，降低噪声
    model.train() # 切回训练模式
    return out # 返回 {'train': ..., 'val': ...}

# 最简单的 bigram（二元）语言模型
class BigramLanguageModel(nn.Module): # 继承 nn.Module 定义模型

    def __init__(self, vocab_size): # 构造函数，传入词表大小
        super().__init__() # 调用父类构造函数
        # 每个 token 直接从查找表中读出下一个 token 的 logits
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # 大小为 (vocab_size, vocab_size) 的嵌入表

    def forward(self, idx, targets=None): # 前向传播，targets 可选

        # idx 和 targets 都是形状为 (B,T) 的整数张量
        logits = self.token_embedding_table(idx) # 查表得到 (B,T,C) 的 logits，C = vocab_size

        if targets is None: # 推理（生成）时没有目标
            loss = None # 不计算 loss
        else: # 训练时有目标
            B, T, C = logits.shape # 取出批大小、序列长度、通道数
            logits = logits.view(B*T, C) # 展平成 (B*T, C)，符合 cross_entropy 的输入要求
            targets = targets.view(B*T) # 目标也展平成 (B*T,)
            loss = F.cross_entropy(logits, targets) # 计算交叉熵损失

        return logits, loss # 返回 logits 和 loss

    def generate(self, idx, max_new_tokens): # 自回归生成新 token
        # idx 是当前上下文的索引，形状为 (B, T)
        for _ in range(max_new_tokens): # 每次循环生成一个新 token
            # 获取预测结果
            logits, loss = self(idx) # 前向传播
            # 只关注最后一个时间步
            logits = logits[:, -1, :] # 取最后一个位置，变为 (B, C)
            # 用 softmax 得到概率分布
            probs = F.softmax(logits, dim=-1) # (B, C)
            # 按概率分布采样
            idx_next = torch.multinomial(probs, num_samples=1) # 采样一个 token，(B, 1)
            # 把采样到的 token 拼接到序列末尾
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx # 返回包含生成内容的完整序列

model = BigramLanguageModel(vocab_size) # 实例化模型
m = model.to(device) # 把模型参数搬到 GPU/CPU 上

# 创建 PyTorch 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # 使用 AdamW 优化器

for iter in range(max_iters): # 训练主循环

    # 每隔一段时间在训练集和验证集上评估 loss
    if iter % eval_interval == 0: # 到了评估的步数
        losses = estimate_loss() # 估计 loss
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}") # 打印训练和验证 loss

    # 采样一个批次的数据
    xb, yb = get_batch('train') # 从训练集取一个批次

    # 计算 loss
    logits, loss = model(xb, yb) # 前向传播
    optimizer.zero_grad(set_to_none=True) # 清空上一步的梯度
    loss.backward() # 反向传播，计算梯度
    optimizer.step() # 根据梯度更新参数

# 用模型生成文本
context = torch.zeros((1, 1), dtype=torch.long, device=device) # 起始上下文：一个值为 0 的 token（换行符）
print(decode(m.generate(context, max_new_tokens=500)[0].tolist())) # 生成 500 个字符，解码后打印
