from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
)
import torch
from typing import Optional
from peft import (
    get_peft_model,
    TaskType,
    LoraConfig,
    AutoPeftModelForCausalLM,
    AutoPeftModelForSeq2SeqLM,
)
from trl import AutoModelForSeq2SeqLMWithValueHead, AutoModelForCausalLMWithValueHead
import transformers
import torch.nn as nn
import argparse


class LoadModels(object):
    def __init__(self, args: argparse.Namespace):
        self.model_id = args.model_name
        self.ref_model_path = args.ref_model_path
        self.peft = args.use_peft
        self.args = args
        args.train_using = args.train_using.lower()
        if self.model_id is not None:
            self.is_encoder_decoder = "t5" in self.model_id.lower()
        elif self.ref_model_path is not None:
            self.is_encoder_decoder = "t5" in self.ref_model_path.lower()

        if self.ref_model_path is None:
            assert (
                self.model_id is not None
            ), "--model-name is required if ref-model-path is not provided - model-name is the huggingface model-id to use"
            self.ref_model_path = self.model_id
            self.model, self.tokenizer = self.load_base_model(self.model_id)
            self.ref_model = None
        else:
            if self.model_id is None:
                self.model_id = self.ref_model_path
            self.model, self.ref_model, self.tokenizer = self.load_fine_tuned_model()

        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.model.enable_input_require_grads()

    def load_base_model(self, model_id: str) -> transformers.PreTrainedModel:
        if self.is_encoder_decoder:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                device_map=None,
                # trust_remote_code=True,
                # low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=None,
                # trust_remote_code=True,
                # low_cpu_mem_usage=True,
            )

        if self.peft:
            # Determine target modules based on model type
            # if "qwen" in model_id.lower():
            #     target_modules = ["q_proj", "v_proj"]
            # else:
            #     # Default target modules for other models
            #     target_modules = ["q_proj", "v_proj"]

            target_modules = "all-linear"

            peft_config = LoraConfig(
                task_type=(
                    TaskType.SEQ2SEQ if self.is_encoder_decoder else TaskType.CAUSAL_LM
                ),
                inference_mode=False,
                r=self.args.peft_config_r,
                lora_alpha=self.args.peft_config_lora_alpha,
                lora_dropout=self.args.peft_config_lora_dropout,
                target_modules=target_modules,
            )

            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            # padding_side="right",
            # padding=True,
            # model_max_length=self.args.max_length,
        )
        return model, tokenizer

    def load_fine_tuned_model(self) -> transformers.PreTrainedModel:
        tokenizer = AutoTokenizer.from_pretrained(self.ref_model_path)
        if self.args.train_using == "ppo":
            if self.is_encoder_decoder:
                model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
                    self.ref_model_path
                ).to("cuda:0")
            else:
                model = AutoModelForCausalLMWithValueHead.from_pretrained(
                    self.ref_model_path
                ).to("cuda:0")
            return model, None, tokenizer
        else:
            if self.args.use_peft:
                cls = (
                    AutoPeftModelForSeq2SeqLM
                    if self.is_encoder_decoder
                    else AutoPeftModelForCausalLM
                )
            else:
                cls = (
                    AutoModelForSeq2SeqLM
                    if self.is_encoder_decoder
                    else AutoModelForCausalLM
                )

        model = cls.from_pretrained(
            self.ref_model_path, device_map="auto", **{"is_trainable": True}
        )
        ref_model = cls.from_pretrained(
            self.ref_model_path, device_map="auto", **{"is_trainable": False}
        )

        return model, ref_model, tokenizer

    def __str__(self):
        return f"Model: {self.model_id}"

    # def load_fine_tuned_model(self) -> transformers.PreTrainedModel:
    #     # tokenizer 优先用基座（如果提供了），否则用适配器目录
    #     tok_src = self.model_id if self.model_id is not None else self.ref_model_path
    #     tokenizer = AutoTokenizer.from_pretrained(tok_src)

    #     if self.args.train_using == "ppo":
    #         if self.is_encoder_decoder:
    #             model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
    #                 self.ref_model_path
    #             ).to("cuda:0")
    #         else:
    #             model = AutoModelForCausalLMWithValueHead.from_pretrained(
    #                 self.ref_model_path
    #             ).to("cuda:0")
    #         return model, None, tokenizer
    #     else:
    #         if self.args.use_peft:
    #             peft_cls = (
    #                 AutoPeftModelForSeq2SeqLM if self.is_encoder_decoder else AutoPeftModelForCausalLM
    #             )
    #             base_cls = (
    #                 AutoModelForSeq2SeqLM if self.is_encoder_decoder else AutoModelForCausalLM
    #             )
    #             try:
    #                 # 情况A：适配器目录包含 base_model_name_or_path，能直接恢复
    #                 model = peft_cls.from_pretrained(self.ref_model_path, device_map="auto", **{"is_trainable": True})
    #                 ref_model = peft_cls.from_pretrained(self.ref_model_path, device_map="auto", **{"is_trainable": False})
    #             except Exception:
    #                 # 情况B：只有适配器，没有记录基座；使用 --model-name 作为基座
    #                 assert self.model_id is not None, "use_peft 时仅有适配器需要提供 --model-name 作为基座模型"
    #                 base = base_cls.from_pretrained(self.model_id, device_map="auto")
    #                 model = PeftModel.from_pretrained(base, self.ref_model_path, is_trainable=True)

    #                 ref_base = base_cls.from_pretrained(self.model_id, device_map="auto")
    #                 ref_model = PeftModel.from_pretrained(ref_base, self.ref_model_path, is_trainable=False)
    #             return model, ref_model, tokenizer
    #         else:
    #             cls = AutoModelForSeq2SeqLM if self.is_encoder_decoder else AutoModelForCausalLM

    #     model = cls.from_pretrained(self.ref_model_path, device_map="auto", **{"is_trainable": True})
    #     ref_model = cls.from_pretrained(self.ref_model_path, device_map="auto", **{"is_trainable": False})

    #     return model, ref_model, tokenizer
