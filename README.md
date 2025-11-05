```markdown
# FTCoT

FTCoT是一个用于论辩对话处理、模型训练与响应优化的工具框架。该框架支持从数据预处理、论辩分析到模型微调（SFT/PPO）的全流程任务，适用于辩论对话生成场景。


## 主要功能

- 数据预处理：支持 Reddit 等对话数据的加载、清洗与格式转换（见 `pre_process/` 目录）。
- 论辩分析：对辩论文本的结构、质量进行自动化分析，并生成关键改进建议（见 `post_process`）。
- 模型训练：基于 Hugging Face Transformers 实现 SFT（监督微调）和 PPO（强化学习微调），支持 LoRA 轻量化训练（见 `train/` 目录）。



## 安装依赖
核心依赖包括：
- `transformers`：模型加载与训练
- `torch`：深度学习框架
- `peft`：参数高效微调（LoRA）
- `trl`：强化学习（PPO）训练工具
- `datasets`：数据处理
- `openai`：调用 LLM API（可选）


## 快速开始

### 1. 数据预处理

以 Reddit 对话数据为例，处理成模型可训练的格式：

```bash
python pre_process/step1_construct_reddit_data.py
```

### 2. 论辩分析

```bash
# 生成论证分析数据
python post_process/argument_analysis/prepare_for_sft.py

# 优化响应
python post_process/response_optimization/prepare_for_sft.py
```

### 3. 模型训练

#### 监督微调（SFT）

```bash
python train/trainer.py --model-name <huggingface-model-id> --output-dir <save-path> --train-using sft
```




## 项目结构

```
FTCoT/
├── pre_process/          # 数据预处理脚本
├── train/                # 模型训练相关（SFT/PPO/模型加载）
├── post_process/         # 后处理（论证分析、响应优化）
│   ├── argument_analysis/
│   └── response_optimization/
├── utils.py              # 通用工具函数（JSON/JSONL 读写等）
├── call_llm.py           # LLM 调用基础类
└── test.py               # LLM 调用示例
```


## 注意事项

- 训练大型模型需配置足够的 GPU 资源，建议使用 DeepSpeed 加速（见 `train/ds_config_zero2.json` 和 `ds_config_zero3.json`）。
- 调用第三方 LLM API 时，需确保网络通畅并替换为有效的 API 密钥。
- 数据格式需符合脚本要求，详细可参考 `pre_process/` 中的示例数据处理逻辑。
```
