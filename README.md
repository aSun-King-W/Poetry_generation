# 古诗生成系统

输入古诗上句，自动生成对应下句。从零实现 **Decoder-Only** 和 **Encoder-Decoder** 两种 Transformer 架构，外加 **Pretrained GPT-2 微调**，共三种方案横向对比。

## 项目结构

```
├── models/
│   ├── transformer.py      # 共享 Transformer 组件（从零实现，不调 nn.Transformer）
│   ├── decoder_only.py     # 方法一：Decoder-Only 架构
│   ├── encoder_decoder.py  # 方法二：Encoder-Decoder 架构
│   └── pretrained.py       # 方法三：uer/gpt2-chinese-poem 微调
├── utils/
│   ├── tokenizer.py        # 数据清洗、词表构建
│   └── dataset.py          # Dataset 封装，三种模型不同数据格式
├── train.py                # 统一训练入口（支持单卡/多卡）
├── generate.py             # 推理生成（贪心 / 集束搜索 / 温度采样）
├── evaluate.py             # 评估（Perplexity + BLEU-4）
├── app.py                  # Gradio Web Demo（三模型对比）
├── data/                   # 数据集 & 词表
├── checkpoints/            # 训练好的模型权重
└── report/                 # LaTeX 报告
```

## 三种模型架构

### 方法一：Decoder-Only Transformer

- 6 层 Transformer Decoder，d_model=256，8 头注意力
- 因果掩码自回归，输入格式 `上句 <SEP> 下句 <EOS>`

### 方法二：Encoder-Decoder Transformer

- 4 层 Encoder + 4 层 Decoder，d_model=256
- Encoder 编码上句，Decoder 通过 Cross-Attention 解码下句

### 方法三：Pretrained Fine-Tuning

- 基于 `uer/gpt2-chinese-poem`（GPT-2 架构，80 万首古诗预训练）
- 小学习率微调，利用了预训练知识

### 超参数对比

| 参数 | Decoder-Only | Encoder-Decoder | Pretrained |
|------|:------------:|:---------------:|:----------:|
| d_model | 256 | 256 | 768 |
| 层数 | 6 | 4+4 | 12 |
| 注意力头 | 8 | 8 | 12 |
| 学习率 | 1e-4 | 1e-4 | 2e-5 |
| Batch Size | 64 | 64 | 16 |
| Epochs | 30 | 30 | 10 |

## 数据集

- 来源：[chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)（唐诗、宋诗 JSON）
- 处理：同一首诗中相邻行配对，筛选五言/七言
- 统计：约 13 万对 (上句, 下句) | 词表 ~6500 | 8:1:1 划分

## 实验结果

| Model | PPL ↓ | BLEU-4 ↑ |
|-------|:-----:|:---------:|
| Decoder-Only | 79.15 | 0.0010 |
| Encoder-Decoder | 102.44 | 0.0010 |
| Pretrained | **1.49** | **0.0136** |

> 注：诗歌生成中一句上句可以有多种合理的下句，BLEU 只测量与单一参考的 n-gram 重合度，分数偏低属正常现象。

## 使用方式

### 训练

```bash
# Decoder-Only
python train.py --model decoder_only --epochs 30 --batch-size 64

# Encoder-Decoder
python train.py --model encoder_decoder --epochs 30 --batch-size 64

# Pretrained 微调
python train.py --model pretrained --epochs 10 --batch-size 16

# 多卡训练
python train.py --model decoder_only --multi-gpu
```

### 推理

```bash
python generate.py --model decoder_only --input "床前明月光"
python generate.py --model decoder_only --input "床前明月光" --strategy beam --beam-size 5
python generate.py --model encoder_decoder --input "举头望明月" --strategy sample --temperature 0.8
```

### 评估

```bash
python evaluate.py --model decoder_only
python evaluate.py --model encoder_decoder
python evaluate.py --model pretrained
```

### Gradio Web Demo

```bash
python app.py
```

打开浏览器访问 Gradio 界面，在输入框输入上句，选择模型和解码策略，一键生成并对比如图。

## 实现要点

- **从零实现 Transformer**：不使用 `nn.Transformer` / `nn.MultiheadAttention`，所有组件（多头注意力、位置编码、LayerNorm、FeedForward、TransformerBlock / EncoderBlock）均用 PyTorch 张量运算手写。
- **Pre-Norm 架构**：每个子层按 Residual → LayerNorm → Sublayer 顺序排列，训练更稳定。
- **混合精度训练**：支持 AMP 自动混合精度，减少显存占用。
- **多卡支持**：`--multi-gpu` 使用 `DataParallel` 多卡并行。
- **Unlikelihood 损失**：Pretrained 模型训练可选 unlikelihood loss 抑制重复。

## 环境要求

- Python ≥ 3.8
- PyTorch ≥ 1.13
- CUDA（可选，推荐用于训练）
