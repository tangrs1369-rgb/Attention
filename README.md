# Attention

复刻 Andrej Karpathy 视频 [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) 中的代码：在 Tiny Shakespeare 数据集上训练一个字符级的 decoder-only Transformer（GPT）。

## 文件

| 文件 | 说明 |
| --- | --- |
| `input.txt` | Tiny Shakespeare 数据集（约 1MB） |
| `bigram.py` | 基线模型：Bigram 语言模型，每个 token 只看前一个 token |
| `gpt.py` | 完整 GPT：多头自注意力 + 前馈网络 + 残差连接 + LayerNorm（pre-norm）+ Dropout |

## gpt.py 的结构（对应视频中逐步搭建的过程）

1. **Head**：单头因果自注意力，`wei = softmax(q·kᵀ / √head_size)`，用下三角 `tril` 掩码遮住未来位置
2. **MultiHeadAttention**：多个头并行，拼接后经过投影层
3. **FeedFoward**：`Linear(C, 4C) → ReLU → Linear(4C, C)`
4. **Block**：`x = x + sa(ln1(x))`，`x = x + ffwd(ln2(x))`（"先通信，再计算"）
5. **GPTLanguageModel**：token embedding + position embedding → N 个 Block → 最终 LayerNorm → lm_head

默认超参数与视频一致：`n_embd=384, n_head=6, n_layer=6, block_size=256, dropout=0.2`，约 10.8M 参数。

## 运行

```bash
pip install torch
python bigram.py   # 几秒钟，val loss ≈ 2.49
python gpt.py      # 视频中在 A100 上约 15 分钟，val loss ≈ 1.48
```

完整的 `gpt.py` 在 CPU 上会非常慢。没有 GPU 的话，可以把超参数调小，例如
`batch_size=16, block_size=32, n_embd=64, n_head=4, n_layer=4, max_iters=1000, learning_rate=1e-3, dropout=0.0`，
这样 CPU 上大约 30 秒就能跑完，val loss 约 2.1，已经能生成像莎士比亚剧本格式的文本。
