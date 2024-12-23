from typing import Any, Mapping, Optional, Union, Sequence, Literal
from globals_dir.globals import GLOBALS
from .instantiate_model import InstantiateModels


class ConfigOptiens:

    def __init__(self):
        self.numa: Optional[bool] = None
        self.num_ctx: Optional[int] = None
        self.num_batch: Optional[int] = None
        self.num_gpu: Optional[int] = None
        self.main_gpu: Optional[int] = None
        self.low_vram: Optional[bool] = None
        self.f16_kv: Optional[bool] = None
        self.logits_all: Optional[bool] = None
        self.vocab_only: Optional[bool] = None
        self.use_mmap: Optional[bool] = None
        self.use_mlock: Optional[bool] = None
        self.embedding_only: Optional[bool] = None
        self.num_thread: Optional[int] = None

        # runtime options
        self.num_keep: Optional[int] = None
        self.seed: Optional[int] = None
        self.num_predict: Optional[int] = None
        self.top_k: Optional[int] = None
        self.top_p: Optional[float] = None
        self.tfs_z: Optional[float] = None
        self.typical_p: Optional[float] = None
        self.repeat_last_n: Optional[int] = None
        self.temperature: Optional[float] = None
        self.repeat_penalty: Optional[float] = None
        self.presence_penalty: Optional[float] = None
        self.frequency_penalty: Optional[float] = None
        self.mirostat: Optional[int] = None
        self.mirostat_tau: Optional[float] = None
        self.mirostat_eta: Optional[float] = None
        self.penalize_newline: Optional[bool] = None
        self.stop: Optional[Sequence[str]] = None


class ModelConfig:
    """download, activate, and manage all the parameters"""

    def __init__(self, model_name, task="null"):
        self.pull: bool = True
        self.local_path: str = ""
        self.modelfile_content: str = ""
        self.to: str = ""
        self.model_name = model_name
        self.task: Literal[*GLOBALS.available_tasks] = task
        self.client = GLOBALS.client

        self.options = ConfigOptiens()
        self.suffix = ''
        self.system = None
        self.template = None
        self.context = None
        self.raw = None
        self.format = None
        self.keep_alive = None

    def init(self):
        self.verify()
        if self.to:
            pass
        if self.pull:
            InstantiateModels.pull(self.model_name)
        elif self.local_path:
            InstantiateModels.create(self.model_name, self.local_path, self.modelfile_content)


    def verify(self):
        # should verify that all the parameters are OK
        pass

