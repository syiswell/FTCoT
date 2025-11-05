import argparse
from trl import (
    SFTConfig,
    DPOConfig,
    CPOConfig,
    KTOConfig,
    SFTTrainer,
    DPOTrainer,
    CPOTrainer,
    KTOTrainer,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-using",
        type=str,
        required=True,
        help="training to perform -- either sft or preference optimization (sft (Instruction Tuning), dpo, cpo, kto, fipo, ppo)",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="/home/sunyang/hlt/DebateTree/log/test/train",
        help="path to json training data - for sft and preference optimization",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="huggingface model-id e.g., meta-llama/Llama-2-7b-hf",
    )

    ## Specific to preference optimization
    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--ref-model-path", default=None)

    ## Specific to PEFT
    parser.add_argument("--peft-config-r", default=16, type=int)
    parser.add_argument("--peft-config-lora-alpha", default=48, type=float)
    parser.add_argument("--peft-config-lora-dropout", default=0.05, type=float)
    parser.add_argument("--use-peft", type=bool, default=True)
    parser.add_argument("--max-length", default=4096, type=int)

    ## specific to KTO
    parser.add_argument("--desirable-weight", default=1.0, type=float)
    parser.add_argument("--undesirable-weight", default=1.0, type=float)

    ## Specific to PPO
    parser.add_argument("--reward-model-path")

    ## Training arguments
    parser.add_argument("--n-epochs", default=2, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--eval-batch-size", default=32, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
    parser.add_argument("--learning-rate", default=1e-5, type=float)
    parser.add_argument("--warmup-ratio", default=0.0, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--adam-epsilon", default=1e-8, type=float)
    parser.add_argument("--save-steps", default=500, type=int)
    parser.add_argument("--logging-steps", default=300, type=int)
    parser.add_argument("--output-dir", default="models", type=str)

    parser.add_argument("--dataloader-num-workers", default=4, type=int)
    parser.add_argument("--dataloader-pin-memory", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    # DeepSpeed arguments
    parser.add_argument(
        "--deepspeed", action="store_true", help="Use DeepSpeed for training"
    )
    parser.add_argument(
        "--deepspeed_config", type=str, default=None, help="DeepSpeed config file path"
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )

    return parser.parse_args()


def get_trainer_and_config_cls(method_used):
    if "sft" in method_used:
        config = SFTConfig
        trainer = SFTTrainer

    elif "dpo" in method_used:
        config = DPOConfig
        trainer = DPOTrainer

    elif "cpo" in method_used:
        config = CPOConfig
        trainer = CPOTrainer

    elif "kto" in method_used:
        config = KTOConfig
        trainer = KTOTrainer

    else:
        raise ValueError(f"Method {method_used} not supported")

    return config, trainer
