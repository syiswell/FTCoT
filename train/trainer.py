from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import deepspeed
from deepspeed import DeepSpeedEngine
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset, Dataset
from data_loader import DataLoader, CustomSFTDataCollator
from model import LoadModels
import torch.nn as nn
import pandas as pd
import argparse
import PO_utils
import pathlib
import torch
import sys
import ppo
import json
import os
import torch.distributed as dist

sys.path.append("train/")


class Trainer:
    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: Optional[PreTrainedModel],
        tokenizer: AutoTokenizer,
        args: argparse.Namespace,
        trainer_cls: Trainer,
        config_cls: TrainingArguments,
        train_dataset: Dataset,
        is_encoder_decoder: bool = False,
        config_kwargs: Dict = {},
        training_kwargs: Dict = {},
    ) -> None:

        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.args = args
        self.trainer_cls = trainer_cls
        self.config_cls = config_cls
        self.train_dataset = train_dataset
        self.config_kwargs = config_kwargs
        self.training_kwargs = training_kwargs

        # 添加 DeepSpeed 配置
        self.deepspeed_config = getattr(args, "deepspeed_config", None)

        self.training_arguments = {
            "output_dir": args.output_dir,
            "overwrite_output_dir": False,
            "num_train_epochs": args.n_epochs,
            "per_device_train_batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "adam_epsilon": args.adam_epsilon,
            "save_steps": args.save_steps,
            # 'eval_steps':4000,
            # 'eval_strategy':'steps',
            "logging_steps": 10,
            "save_total_limit": 2,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            # 多GPU和性能优化参数
            "dataloader_num_workers": args.dataloader_num_workers,
            "dataloader_pin_memory": args.dataloader_pin_memory,
            "bf16": args.bf16,  # 使用 bf16 减少显存
            "gradient_checkpointing": args.gradient_checkpointing,  # 梯度检查点
            "eval_strategy": "no",
            "report_to": "none",  # 关闭 wandb 以减少开销
            # 添加 DeepSpeed 相关参数
            "deepspeed": self.deepspeed_config,
            "local_rank": getattr(args, "local_rank", -1),
            "ddp_find_unused_parameters": False,  # 提高性能
            "dataloader_drop_last": True,  # 避免最后一个不完整的batch
        }

        self.sft_args = {
            "max_seq_length": args.max_length,
        }

        self.preference_optimization_args = {
            "beta": args.beta,
            "max_prompt_length": args.max_length,
            "max_length": args.max_length,
            # "max_target_length": 512,
            "is_encoder_decoder": is_encoder_decoder,
            "generate_during_eval": True,
        }

        self.config, self.trainer = self.init_config_trainer()

    def init_config_trainer(self) -> Tuple[TrainingArguments, Trainer]:
        if "sft" not in self.args.train_using:
            config = self.config_cls(
                **self.training_arguments,
                **self.preference_optimization_args,
                **self.config_kwargs,
            )
            # 添加 DeepSpeed 配置
            if self.deepspeed_config:
                config.deepspeed = self.deepspeed_config

            trainer = self.trainer_cls(
                **{
                    "model": self.model,
                    "ref_model": self.ref_model,
                    "tokenizer": self.tokenizer,
                    "args": config,
                    "train_dataset": self.train_dataset,
                },
                **self.training_kwargs,
            )
        else:
            config = self.config_cls(
                **self.training_arguments, **self.sft_args, **self.config_kwargs
            )

            # 添加 DeepSpeed 配置
            if self.deepspeed_config:
                config.deepspeed = self.deepspeed_config

            # 创建自定义的data collator
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                padding="longest",  # 显式启用填充
                # max_length=self.args.max_length,  # 设置最大长度
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
            # data_collator = CustomSFTDataCollator(self.tokenizer)

            trainer = self.trainer_cls(
                **{
                    "model": self.model,
                    "tokenizer": self.tokenizer,
                    "args": config,
                    "train_dataset": self.train_dataset,
                    "data_collator": data_collator,
                },
                **self.training_kwargs,
            )
        return config, trainer

    def train_(self) -> None:
        self.trainer.train()

        # 只在主进程保存模型
        if not dist.is_initialized() or dist.get_rank() == 0:
            self.trainer.save_model(self.args.output_dir)


def main() -> None:
    # 初始化分布式训练
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    args = PO_utils.parse_args()
    args.train_use = args.train_using.lower()

    # 设置 local_rank
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    # 将超参保存到 output_dir/args.json（新增）
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    if args.train_using == "ppo":
        ppo.train(args)
        return

    loaded_models = LoadModels(args=args)
    if loaded_models.tokenizer.pad_token is None:
        loaded_models.tokenizer.pad_token = loaded_models.tokenizer.eos_token

    data_loader = DataLoader(args, loaded_models.tokenizer)
    data_loader.load_data(args.train_using)

    config_cls, trainer_cls = PO_utils.get_trainer_and_config_cls(args.train_using)
    TRAINER = Trainer(
        model=loaded_models.model,
        ref_model=loaded_models.ref_model,
        tokenizer=loaded_models.tokenizer,
        args=args,
        trainer_cls=trainer_cls,
        config_cls=config_cls,
        is_encoder_decoder=loaded_models.is_encoder_decoder,
        train_dataset=data_loader.__getdata__(),
        # config_kwargs=data_loader.__configkwargs__(),
        # training_kwargs=data_loader.__trainerkwargs__(),
    )

    TRAINER.train_()


if __name__ == "__main__":
    main()
