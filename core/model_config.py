from typing import Any, Mapping, Optional, Union, Sequence, Literal
from globals_dir.globals import GLOBALS


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

    # def __getattr__(self, name):
    #     if name in self._attributes:
    #         return self._attributes[name]
    #     raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    #
    # # Generic setter with type enforcement
    # def __setattr__(self, name, value):
    #     if name in ("_attributes", "_types"):
    #         super().__setattr__(name, value)
    #     elif name in self._attributes:
    #         expected_type = self._types[name]
    #         if not self._validate_type(value, expected_type):
    #             raise TypeError(f"{name} must be of type {expected_type} (got {type(value).__name__})")
    #         self._attributes[name] = value
    #     else:
    #         raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    #
    # def _validate_type(self, value, expected_type):
    #     if expected_type is None:
    #         return True
    #     # Handle Optional[Type]
    #     origin = getattr(expected_type, "__origin__", None)
    #     if origin is Union and type(None) in expected_type.__args__:
    #         return isinstance(value, expected_type.__args__) or value is None
    #     # Direct check
    #     return isinstance(value, expected_type)


class ModelConfig:
    """download, activate, and manage all the parameters"""

    def __init__(self, model_name, task="null"):
        self.pull: bool = True
        self.loc: str = ""
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

    def instantiate(self):
        if self.to:
            pass
        if self.pull:
            pass
        elif self.loc:
            pass

    def verify(self):
        # should verify that all the parameters are OK
        pass

    # def __getattr__(self, name):
    #     if name in self._attributes:
    #         return self._attributes[name]
    #     raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    #
    # # Generic setter with type enforcement
    # def __setattr__(self, name, value):
    #     if name in ("_attributes", "_types"):
    #         super().__setattr__(name, value)
    #     elif name in self._attributes:
    #         expected_type = self._types[name]
    #         if not self._validate_type(value, expected_type):
    #             raise TypeError(f"{name} must be of type {expected_type} (got {type(value).__name__})")
    #         self._attributes[name] = value
    #     else:
    #         raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    #
    # def _validate_type(self, value, expected_type):
    #     if expected_type is None:
    #         return True
    #     # Handle Optional[Type]
    #     origin = getattr(expected_type, "__origin__", None)
    #     if origin is Union and type(None) in expected_type.__args__:
    #         return isinstance(value, expected_type.__args__) or value is None
    #     # Direct check
    #     return isinstance(value, expected_type)
