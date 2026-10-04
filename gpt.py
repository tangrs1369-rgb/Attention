import torch # 导入 PyTorch 主库
import torch.nn as nn # 导入神经网络模块，简写为 nn
from torch.nn import functional as F # 导入函数式接口（softmax、cross_entropy 等），简写为 F

# 超参数
batch_size = 64 # 每个批次并行处理多少条独立的序列
block_size = 256 # 预测时最多能看到的上下文长度（序列长度）
max_iters = 5000 # 训练总步数
eval_interval = 500 # 每隔多少步评估一次 loss
learning_rate = 3e-4 # 学习率（模型变大后要用较小的学习率）
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 有 GPU 就用 GPU，否则用 CPU
eval_iters = 200 # 评估 loss 时取多少个批次求平均
n_embd = 384 # 嵌入维度（每个 token 的向量长度）
n_head = 6 # 注意力头的数量，每个头的维度为 384/6 = 64
n_layer = 6 # Transformer Block 的层数
dropout = 0.2 # dropout 比例，每次前向随机丢弃 20% 的神经元，防止过拟合
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
    model.eval() # 切换到评估模式（关闭 dropout）
    for split in ['train', 'val']: # 分别评估训练集和验证集
        losses = torch.zeros(eval_iters) # 用来记录每个批次的 loss
        for k in range(eval_iters): # 取 eval_iters 个批次
            X, Y = get_batch(split) # 取一个批次
            logits, loss = model(X, Y) # 前向传播得到 loss
            losses[k] = loss.item() # 记录该批次的 loss（转成 Python 数值）
        out[split] = losses.mean() # 求平均，降低噪声
    model.train() # 切回训练模式（重新开启 dropout）
    return out # 返回 {'train': ..., 'val': ...}

class Head(nn.Module): # 单个自注意力头
    """ 一个自注意力头 """

    def __init__(self, head_size): # 构造函数，head_size 为该头的维度
        super().__init__() # 调用父类构造函数
        self.key = nn.Linear(n_embd, head_size, bias=False) # key 投影：“我包含什么信息”
        self.query = nn.Linear(n_embd, head_size, bias=False) # query 投影：“我在找什么信息”
        self.value = nn.Linear(n_embd, head_size, bias=False) # value 投影：“如果你关注我，我提供什么”
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # 下三角矩阵，作为因果掩码；注册为 buffer（不是可训练参数）

        self.dropout = nn.Dropout(dropout) # 对注意力权重做 dropout

    def forward(self, x): # 前向传播
        # 输入形状 (batch, time-step, channels)
        # 输出形状 (batch, time-step, head size)
        B,T,C = x.shape # 取出批大小、序列长度、通道数
        k = self.key(x)   # 计算 key，(B,T,hs)
        q = self.query(x) # 计算 query，(B,T,hs)
        # 计算注意力分数（“亲和度”）
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # q 与 k 点积并除以 √hs 缩放，(B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # 把未来位置填成 -inf，使每个 token 只能看到自己和之前的 token，(B, T, T)
        wei = F.softmax(wei, dim=-1) # 按行 softmax 归一化成权重，-inf 处变为 0，(B, T, T)
        wei = self.dropout(wei) # 随机丢弃部分注意力连接
        # 用注意力权重对 value 做加权聚合
        v = self.value(x) # 计算 value，(B,T,hs)
        out = wei @ v # 加权求和，(B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out # 返回该头的输出

class MultiHeadAttention(nn.Module): # 多头注意力
    """ 多个自注意力头并行运行 """

    def __init__(self, num_heads, head_size): # 构造函数：头的数量和每个头的维度
        super().__init__() # 调用父类构造函数
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # 创建 num_heads 个注意力头
        self.proj = nn.Linear(head_size * num_heads, n_embd) # 输出投影层，把拼接结果映射回残差通路
        self.dropout = nn.Dropout(dropout) # 对输出做 dropout

    def forward(self, x): # 前向传播
        out = torch.cat([h(x) for h in self.heads], dim=-1) # 每个头分别计算，在通道维拼接，(B,T,num_heads*hs)
        out = self.dropout(self.proj(out)) # 投影后做 dropout，(B,T,n_embd)
        return out # 返回多头注意力的输出

class FeedFoward(nn.Module): # 前馈网络
    """ 一个简单的线性层，后接非线性激活 """

    def __init__(self, n_embd): # 构造函数，传入嵌入维度
        super().__init__() # 调用父类构造函数
        self.net = nn.Sequential( # 按顺序组合以下几层
            nn.Linear(n_embd, 4 * n_embd), # 升维到 4 倍（与《Attention Is All You Need》论文一致）
            nn.ReLU(), # ReLU 非线性激活
            nn.Linear(4 * n_embd, n_embd), # 投影回原维度，进入残差通路
            nn.Dropout(dropout), # dropout 防止过拟合
        ) # Sequential 结束

    def forward(self, x): # 前向传播
        return self.net(x) # 每个 token 独立地经过前馈网络（“各自思考”）

class Block(nn.Module): # Transformer 块
    """ Transformer 块：先通信（注意力），再计算（前馈） """

    def __init__(self, n_embd, n_head): # 构造函数
        # n_embd：嵌入维度，n_head：想要的注意力头数量
        super().__init__() # 调用父类构造函数
        head_size = n_embd // n_head # 每个头的维度
        self.sa = MultiHeadAttention(n_head, head_size) # 多头自注意力层
        self.ffwd = FeedFoward(n_embd) # 前馈网络层
        self.ln1 = nn.LayerNorm(n_embd) # 注意力之前的 LayerNorm
        self.ln2 = nn.LayerNorm(n_embd) # 前馈之前的 LayerNorm

    def forward(self, x): # 前向传播
        x = x + self.sa(self.ln1(x)) # 先 LayerNorm 再注意力（pre-norm），并加上残差连接
        x = x + self.ffwd(self.ln2(x)) # 先 LayerNorm 再前馈，并加上残差连接
        return x # 返回该块的输出

class GPTLanguageModel(nn.Module): # 完整的 GPT 语言模型

    def __init__(self): # 构造函数
        super().__init__() # 调用父类构造函数
        # 每个 token 从查找表中读出自己的嵌入向量
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # token 嵌入表，(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # 位置嵌入表，每个位置一个向量，(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # 堆叠 n_layer 个 Transformer 块
        self.ln_f = nn.LayerNorm(n_embd) # 最后的 LayerNorm
        self.lm_head = nn.Linear(n_embd, vocab_size) # 语言模型输出头：把嵌入映射成词表大小的 logits

        # 更好的初始化方式，原 GPT 视频没讲，但很重要，会在后续视频中讲解
        self.apply(self._init_weights) # 对所有子模块应用初始化函数

    def _init_weights(self, module): # 权重初始化函数
        if isinstance(module, nn.Linear): # 如果是线性层
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # 权重用均值 0、标准差 0.02 的正态分布初始化
            if module.bias is not None: # 如果有偏置
                torch.nn.init.zeros_(module.bias) # 偏置初始化为 0
        elif isinstance(module, nn.Embedding): # 如果是嵌入层
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # 同样用标准差 0.02 的正态分布初始化

    def forward(self, idx, targets=None): # 前向传播，targets 可选
        B, T = idx.shape # 取出批大小和序列长度

        # idx 和 targets 都是形状为 (B,T) 的整数张量
        tok_emb = self.token_embedding_table(idx) # token 嵌入，(B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # 位置 0..T-1 的位置嵌入，(T,C)
        x = tok_emb + pos_emb # 两者相加（位置嵌入按批广播），(B,T,C)
        x = self.blocks(x) # 依次经过所有 Transformer 块，(B,T,C)
        x = self.ln_f(x) # 最后的 LayerNorm，(B,T,C)
        logits = self.lm_head(x) # 映射为下一个 token 的 logits，(B,T,vocab_size)

        if targets is None: # 推理（生成）时没有目标
            loss = None # 不计算 loss
        else: # 训练时有目标
            B, T, C = logits.shape # 取出批大小、序列长度、词表大小
            logits = logits.view(B*T, C) # 展平成 (B*T, C)，符合 cross_entropy 的输入要求
            targets = targets.view(B*T) # 目标也展平成 (B*T,)
            loss = F.cross_entropy(logits, targets) # 计算交叉熵损失

        return logits, loss # 返回 logits 和 loss

    def generate(self, idx, max_new_tokens): # 自回归生成新 token
        # idx 是当前上下文的索引，形状为 (B, T)
        for _ in range(max_new_tokens): # 每次循环生成一个新 token
            # 把 idx 裁剪到最后 block_size 个 token（位置嵌入只有 block_size 个）
            idx_cond = idx[:, -block_size:] # 截取最近的上下文
            # 获取预测结果
            logits, loss = self(idx_cond) # 前向传播
            # 只关注最后一个时间步
            logits = logits[:, -1, :] # 取最后一个位置，变为 (B, C)
            # 用 softmax 得到概率分布
            probs = F.softmax(logits, dim=-1) # (B, C)
            # 按概率分布采样
            idx_next = torch.multinomial(probs, num_samples=1) # 采样一个 token，(B, 1)
            # 把采样到的 token 拼接到序列末尾
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx # 返回包含生成内容的完整序列

model = GPTLanguageModel() # 实例化模型
m = model.to(device) # 把模型参数搬到 GPU/CPU 上
# 打印模型参数量
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters') # 统计所有参数个数，以百万（M）为单位

# 创建 PyTorch 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # 使用 AdamW 优化器

for iter in range(max_iters): # 训练主循环

    # 每隔一段时间在训练集和验证集上评估 loss
    if iter % eval_interval == 0 or iter == max_iters - 1: # 到了评估的步数，或者是最后一步
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
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist())) # （可选）生成 10000 个字符写入 more.txt
