from enum import Enum
from typing import Union, Sequence, Optional, Literal, Mapping, Any
from pydantic.json_schema import JsonSchemaValue
from pydantic import BaseModel
import ollama
from src.funcs import *
from src.utils import *


class Role(Enum):
    USER = 1
    ASSISTANT = 2

from src.core._types import Options
###########################

class Model:

    def __init__(
            self,
            model_name,
            docker=False,
            suffix: str = '',
            *,
            system: str = '',
            template: str = '',
            context: Optional[Sequence[int]] = None,
            stream: Literal[True] = True,
            raw: bool = False,
            format: Optional[Union[Literal['', 'json'], JsonSchemaValue]] = None,
            images: Optional[Sequence[Union[str, bytes]]] = None,
            options: Optional[Union[Mapping[str, Any], Options]] = None,
            keep_alive: Optional[Union[float, str]] = None,
    ):
        self.model_name = model_name
        self.logs = []

        self.docker = docker
        self.suffix: suffix
        self.system = system
        self.template = template
        self.context = context
        self.stream = stream
        self.raw = raw
        self.format = format
        self.images = images
        self.options = options
        self.keep_alive = keep_alive


    def init_ollama(self):
        pass


    def chat_stream(self, messages: list[dict]):
        response = ''
        for part in ollama.chat(self.model_name, messages=messages, stream=True):
            response += part['message']['content']
            print(part['message']['content'], end='', flush=True)
        return response

    def chat(self, messages: list[dict]):
        response = ollama.chat(
            model=self.model_name,
            messages=messages
        )

        self.logs.append(response)
        return response.message.content

    def embed(self, query: Union[str, Sequence[str]]):
        return ollama.embed(model=self.model_name, input=query)

    def _generate(self):

    def infer(self, query):
        # overwrite
        pass

    def __lshift__(self, query):
        return self.infer(query)


############################

class Chat(Model):

    def __init__(self, model_name, prompt=None):
        super().__init__(model_name)
        self.messages = []
        self.prompt = prompt
        self._init_prompt()

    def _init_prompt(self):
        if self.prompt:
            self.messages.append({'role': 'assistant', 'content': self.prompt})

    def _add_step(self, role: Role, content: str):
        message = {
            "role": role.name.lower(),
            "content": content
        }
        self.messages.append(message)

    def infer(self, query) -> str:
        self._add_step(role=Role.USER, content=query)
        response = self.chat(messages=self.messages)
        self._add_step(role=Role.ASSISTANT, content=response)
        return response


############################

class Embed(Model):

    def __init__(self, model_name, prefix=None):
        super().__init__(model_name)
        self.messages = []
        self.prefix = prefix

    def infer(self, query: Union[str, Sequence[str]]) -> list[list]:
        return self.embed(query)


class Generate(Model):

    def __init__(
        self,
        model_name,
        prompt,
        stream=False,
        num_keep=5,
        seed=42,
        num_predict=512,
        top_k=20,
        top_p=0.9,
        min_p=0.0,
        typical_p=0.7,
        repeat_last_n=33,
        temperature=0.8,
        repeat_penalty=1.2,
        presence_penalty=1.5,
        frequency_penalty=1.0,
        mirostat=1,
        mirostat_tau=0.8,
        mirostat_eta=0.6,
        penalize_newline=True,
        stop=("\n", "user:"),
        numa=False,
        num_ctx=1024,
        num_batch=2,
        num_gpu=1,
        main_gpu=0,
        low_vram=False,
        vocab_only=False,
        use_mmap=True,
        use_mlock=False,
        num_thread=8
    ):
        self.model_name = model_name
        self.prompt = prompt
        self.stream = stream
        self.num_keep = num_keep
        self.seed = seed
        self.num_predict = num_predict
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.typical_p = typical_p
        self.repeat_last_n = repeat_last_n
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.mirostat = mirostat
        self.mirostat_tau = mirostat_tau
        self.mirostat_eta = mirostat_eta
        self.penalize_newline = penalize_newline
        self.stop = stop
        self.numa = numa
        self.num_ctx = num_ctx
        self.num_batch = num_batch
        self.num_gpu = num_gpu
        self.main_gpu = main_gpu
        self.low_vram = low_vram
        self.vocab_only = vocab_only
        self.use_mmap = use_mmap
        self.use_mlock = use_mlock
        self.num_thread = num_thread


    def infer(self, query):

    