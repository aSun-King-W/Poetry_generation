# 古诗生成系统

输入古诗上句，自动生成对应下句。从零实现 **Decoder-Only** 和 **Encoder-Decoder** 两种 Transformer 架构，外加 **Pretrained GPT-2 微调**，共三种方案横向对比。

## 项目结构

```
├── src/
│   ├── models/
│   │   ├── transformer.py      # 共享 Transformer 组件（从零实现，不调 nn.Transformer）
│   │   ├── decoder_only.py     # 方法一：Decoder-Only 架构
│   │   ├── encoder_decoder.py  # 方法二：Encoder-Decoder 架构
│   │   └── pretrained.py       # 方法三：uer/gpt2-chinese-poem 微调
│   ├── utils/
│   │   ├── tokenizer.py        # 数据清洗、词表构建
│   │   └── dataset.py          # Dataset 封装，三种模型不同数据格式
│   ├── train.py                # 统一训练入口（支持单卡/多卡）
│   ├── generate.py             # 推理生成（贪心 / 集束搜索 / 温度采样）
│   ├── evaluate.py             # 评估（Perplexity + BLEU-4）
│   └── app.py                  # Gradio Web Demo（三模型对比）
├── docs/
│   ├── 计划.md                 # 项目计划与方案
│   ├── 方案.md                 # 详细实现方案
│   ├── 任务要求.md             # 原始任务要求
│   ├── 所遇困难及解决方案.md   # 开发过程中的问题记录
│   └── results.md              # 详细实验结果
├── data/                       # 数据集 & 词表
├── checkpoints/                # 训练好的模型权重
└── report/                     # LaTeX 报告（中英文）
```

## 三种模型架构

### 方法一：Decoder-Only Transformer

- 6 层 Transformer Decoder，d_model=256，8 头注意力，参数量 11M
- 因果掩码自回归，输入格式 `上句 <SEP> 下句 <EOS>`

### 方法二：Encoder-Decoder Transformer

- 4 层 Encoder + 4 层 Decoder，d_model=256，参数量 12M
- Encoder 编码上句，Decoder 通过 Cross-Attention 解码下句

### 方法三：Pretrained Fine-Tuning

- 基于 `uer/gpt2-chinese-poem`（GPT-2 架构，80 万首古诗预训练，~102M 参数）
- 小学习率全参数微调

## 数据集

- 来源：[chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)（全唐诗、宋诗）
- 处理：同一首诗中相邻行配对，筛选五言/七言
- 统计：**1,037,826 对**（全部五言）| 词表 9,119 | 8:1:1 划分

## 实验结果

| Model | PPL ↓ | BLEU-4 ↑ |
|-------|:-----:|:---------:|
| Decoder-Only (from scratch) | 79.15 | 0.0010 |
| Encoder-Decoder (from scratch) | 102.44 | 0.0010 |
| Pretrained GPT-2 (fine-tuned) | **1.49** | **0.0136** |

> 注：诗歌生成中一句上句可以有多种合理的下句，BLEU 只测量与单一参考的 n-gram 重合度，分数偏低属正常现象。

## 使用方式

所有命令在项目根目录执行：

### 训练

```bash
# Decoder-Only
python src/train.py decoder-only

# Encoder-Decoder
python src/train.py encoder-decoder

# Pretrained 微调
python src/train.py pretrained

# 多卡训练
python src/train.py decoder-only --multi-gpu
```

### 推理

```bash
python src/generate.py --model decoder-only --upper "床前明月光"
python src/generate.py --model decoder-only --upper "床前明月光" --method beam --beam-size 5
python src/generate.py --model encoder-decoder --upper "举头望明月" --method sample --temperature 0.8
```

### 评估

```bash
python src/evaluate.py --models decoder-only encoder-decoder pretrained
```

### Gradio Web Demo

```bash
python src/app.py
```

打开浏览器访问 Gradio 界面，在输入框输入上句，选择模型和解码策略，一键生成并对比如图。

## 实现要点

- **从零实现 Transformer**：不使用 `nn.Transformer` / `nn.MultiheadAttention`，所有组件（多头注意力、位置编码、LayerNorm、FeedForward、TransformerBlock / EncoderBlock）均用 PyTorch 张量运算手写。
- **Pre-Norm 架构**：每个子层按 LayerNorm → Sublayer → Residual 顺序排列（Pre-Norm），训练更稳定。
- **混合精度训练**：支持 AMP 自动混合精度，减少显存占用。
- **多卡支持**：`--multi-gpu` 使用 `DataParallel` 多卡并行。

## 环境要求

- Python ≥ 3.8
- PyTorch ≥ 1.13
- CUDA（可选，推荐用于训练）
