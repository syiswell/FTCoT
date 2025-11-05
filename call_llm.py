try:
    from typing import List, Union, Optional, Literal
except:
    from typing import List, Union, Optional
    from typing_extensions import Literal
import dataclasses

from git import Object
import httpx
from vllm import LLM, SamplingParams
import random
import numpy as np
import os
import json

timeout = httpx.Timeout(100.0)

from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
import openai

_api_key = os.getenv("OPENAI_API_KEY")
_url = os.getenv("OPENAI_BASE_URL")

print(_api_key, "\n", _url)

if _url:
    client = openai.OpenAI(timeout=60, api_key=_api_key, base_url=_url)
else:
    client = openai.OpenAI(timeout=60, api_key=_api_key)
MessageRole = Literal["system", "user", "assistant"]


@dataclasses.dataclass()
class Message:
    role: MessageRole
    content: str


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gpt_completion(
    model: str,
    prompt: str,
    max_tokens: int = 1024,
    stop_strs: Optional[List[str]] = None,
    temperature: float = 0.0,
    num_comps=1,
) -> Union[List[str], str]:
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
        n=num_comps,
        request_timeout=120,
    )
    if num_comps == 1:
        return response.choices[0].text  # type: ignore

    return [choice.text for choice in response.choices]  # type: ignore


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(10))
def gpt_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    num_comps=1,
    response_format: Optional[dict] = None,
) -> Union[List[str], str]:

    messages = [dataclasses.asdict(message) for message in messages]
    # messages[0]["content"] = (
    #     "\nWrap any code snippet with a pair of ``` code fences in your response.\n"
    #     + messages[0]["content"]
    # )

    if response_format:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=num_comps,
            response_format=response_format,
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=num_comps,
            # request_timeout=120,
        )
    if num_comps == 1:
        return response.choices[0].message.content  # type: ignore
    return [choice.message.content for choice in response.choices]  # type: ignore


class ModelBase:
    def __init__(self, name: str):
        self.name = name
        self.is_chat = False

    def __repr__(self) -> str:
        return f"{self.name}"

    def generate_chat(
        self,
        messages: List[Message],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        num_comps: int = 1,
        response_format: Union[dict, str] = None,
    ) -> Union[List[str], str]:
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        stop_strs: Optional[List[str]] = None,
        temperature: float = 0.0,
        num_comps=1,
    ) -> Union[List[str], str]:
        raise NotImplementedError


class GPTChat(ModelBase):
    def __init__(self, model_name: str, cache=False):
        self.name = model_name
        self.is_chat = True
        self.cache = cache
        self.cache_dir = f"log/llm_cache/{model_name}"

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def get_cache(self, messages):
        cache_file = os.path.join(self.cache_dir, f"{hash(str(messages))}.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                return json.load(f)
        return None

    def save_cache(self, messages, response):
        cache_file = os.path.join(self.cache_dir, f"{hash(str(messages))}.json")
        with open(cache_file, "w") as f:
            json.dump(response, f)

    def generate_chat(
        self,
        messages: List[Message],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        num_comps: int = 1,
        response_format: Optional[dict] = None,
    ) -> str:
        # messages[0]["content"] = "\nWrap any code snippet with a pair of ``` code fences in your response.\n" + messages[0]["content"]

        if self.cache:
            cache = self.get_cache(messages)
            if cache:
                return cache
        try:
            response = gpt_chat(
                self.name, messages, max_tokens, temperature, num_comps, response_format
            )
        except Exception as e:
            print(f"gpt_chat error: {e}")
            raise e
        if self.cache:
            self.save_cache(messages, response)
        return response


class GPT4(GPTChat):
    def __init__(self):
        super().__init__("gpt-4")


class GPT4Omini(GPTChat):
    def __init__(self):
        super().__init__("gpt-4o-mini")


class GPT4O(GPTChat):
    def __init__(self):
        super().__init__("gpt-4o")


class GPT35(GPTChat):
    def __init__(self):
        super().__init__("gpt-3.5-turbo")


class LLama31_70B(GPTChat):
    def __init__(self):
        super().__init__("meta-llama/Meta-Llama-3.1-70B-Instruct")


class Qwen25_72B(GPTChat):
    def __init__(self):
        super().__init__("Qwen/Qwen2.5-72B-Instruct")


class GPTDavinci(ModelBase):
    def __init__(self, model_name: str):
        self.name = model_name

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        stop_strs: Optional[List[str]] = None,
        temperature: float = 0,
        num_comps=1,
    ) -> Union[List[str], str]:
        return gpt_completion(
            self.name, prompt, max_tokens, stop_strs, temperature, num_comps
        )


class HFModelBase(ModelBase):
    """
    Base for huggingface chat models
    """

    def __init__(self, model_name: str, model, tokenizer, eos_token_id=None):
        self.name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = (
            eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        )
        self.is_chat = True

    def generate_chat(
        self,
        messages: List[Message],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        num_comps: int = 1,
    ) -> Union[List[str], str]:
        # NOTE: HF does not like temp of 0.0.
        if temperature < 0.0001:
            temperature = 0.0001

        prompt = self.prepare_prompt(messages)

        outputs = self.model.generate(
            prompt,
            max_new_tokens=min(max_tokens, self.model.config.max_position_embeddings),
            use_cache=True,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            eos_token_id=self.eos_token_id,
            num_return_sequences=num_comps,
            request_timeout=120,
        )

        outs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        assert isinstance(outs, list)
        for i, out in enumerate(outs):
            assert isinstance(out, str)
            outs[i] = self.extract_output(out)

        if len(outs) == 1:
            return outs[0]  # type: ignore
        else:
            return outs  # type: ignore

    def prepare_prompt(self, messages: List[Message]):
        raise NotImplementedError

    def extract_output(self, output: str) -> str:
        raise NotImplementedError


class StarChat(HFModelBase):
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceH4/starchat-beta",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceH4/starchat-beta",
        )
        super().__init__("starchat", model, tokenizer, eos_token_id=49155)

    def prepare_prompt(self, messages: List[Message]):
        prompt = ""
        for i, message in enumerate(messages):
            prompt += f"<|{message.role}|>\n{message.content}\n<|end|>\n"
            if i == len(messages) - 1:
                prompt += "<|assistant|>\n"

        return self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

    def extract_output(self, output: str) -> str:
        out = output.split("<|assistant|>")[1]
        if out.endswith("<|end|>"):
            out = out[: -len("<|end|>")]

        return out


class myLLM(ModelBase):
    def __init__(self, model_name: str):
        self.name = model_name
        self.is_chat = True
        self.model = LLM(model_name, max_model_len=8192, gpu_memory_utilization=1.0)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_chat(
        self,
        messages: List[Message],
        max_tokens: int = 4096,
        temperature: float = 0.2,
        num_comps: int = 1,
    ) -> Union[List[str], str]:
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # print(temperature)
        sampling_params = SamplingParams(
            temperature=temperature, top_p=0.95, max_tokens=max_tokens
        )
        messages = [dataclasses.asdict(message) for message in messages]
        messages[0]["content"] = (
            "Wrap any code snippet with a pair of ``` code fences in your response.\n"
            + messages[0]["content"]
        )
        # formatted_prompt =  self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        output = self.model.chat(messages, sampling_params=sampling_params)
        random.seed()
        np.random.seed()

        def print_outputs(outputs):
            res = []
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                res.append(generated_text)
            return "\n".join(res)

        return print_outputs(output)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        stop_strs: Optional[List[str]] = None,
        temperature: float = 0.0,
        num_comps=1,
    ) -> Union[List[str], str]:
        raise NotImplementedError


class CodeLlama(HFModelBase):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    def __init__(self, version: Literal["34b", "13b", "7b"] = "34b"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            f"codellama/CodeLlama-{version}-Instruct-hf",
            add_eos_token=True,
            add_bos_token=True,
            padding_side="left",
        )
        model = AutoModelForCausalLM.from_pretrained(
            f"codellama/CodeLlama-{version}-Instruct-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        super().__init__("codellama", model, tokenizer)

    def prepare_prompt(self, messages: List[Message]):
        if messages[0].role != "system":
            messages = [
                Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)
            ] + messages
        messages = [
            Message(
                role=messages[1].role,
                content=self.B_SYS
                + messages[0].content
                + self.E_SYS
                + messages[1].content,
            )
        ] + messages[2:]
        assert all([msg.role == "user" for msg in messages[::2]]) and all(
            [msg.role == "assistant" for msg in messages[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        messages_tokens: List[int] = sum(
            [
                self.tokenizer.encode(
                    f"{self.B_INST} {(prompt.content).strip()} {self.E_INST} {(answer.content).strip()} ",
                )
                for prompt, answer in zip(
                    messages[::2],
                    messages[1::2],
                )
            ],
            [],
        )
        assert (
            messages[-1].role == "user"
        ), f"Last message must be from user, got {messages[-1].role}"
        messages_tokens += self.tokenizer.encode(
            f"{self.B_INST} {(messages[-1].content).strip()} {self.E_INST}",
        )
        # remove eos token from last message
        messages_tokens = messages_tokens[:-1]
        import torch

        return torch.tensor([messages_tokens]).to(self.model.device)

    def extract_output(self, output: str) -> str:
        out = output.split("[/INST]")[-1].split("</s>")[0].strip()
        return out


def model_factory(model_name: str) -> ModelBase:
    if model_name == "gpt-4":
        return GPT4()
    elif "gpt-3.5-turbo" in model_name:
        return GPT35()
    elif "gpt-4o-mini" in model_name:
        return GPT4Omini()
    elif "gpt-4o" in model_name:
        return GPT4O()
    elif "llama-3.1-70B" in model_name:
        return LLama31_70B()
    elif "qwen2.5-72B" in model_name:
        return Qwen25_72B()
    else:
        return GPTChat(model_name)
    # elif model_name == "starchat":
    #     return StarChat()
    # elif model_name.startswith("codellama"):
    #     # if it has `-` in the name, version was specified
    #     kwargs = {}
    #     if "-" in model_name:
    #         kwargs["version"] = model_name.split("-")[1]
    #     return CodeLlama(**kwargs)
    # elif model_name.startswith("text-davinci"):
    #     return GPTDavinci(model_name)
    # else:
    #     return myLLM(model_name)
    #     # raise ValueError(f"Invalid model name: {model_name}")
