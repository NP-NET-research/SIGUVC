from typing import Dict
from .instructions import *


class PromptTemplate:
    
    def __init__(self, system: str, instruction: str, functions=None):
        self.system = system
        self.instruction = instruction
        self.functions = functions or []

    def build(self, tokenizer, query: str) -> str:
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.instruction + query}
        ]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            functions=self.functions
        )


class PromptRegistry:
    _registry: Dict[str, PromptTemplate] = {}

    @classmethod
    def register(cls, name: str, template: PromptTemplate):
        """注册模板"""
        cls._registry[name] = template

    @classmethod
    def get(cls, name: str) -> PromptTemplate:
        """获取模板"""
        if name not in cls._registry:
            raise KeyError(f"PromptTemplate {name} not found")
        return cls._registry[name]

    @classmethod
    def list_templates(cls):
        """列出所有模板"""
        return list(cls._registry.keys())

    @classmethod
    def replace(cls, name: str, new_template: PromptTemplate):
        """替换模板"""
        cls._registry[name] = new_template

PromptRegistry.register(
    "base",
    PromptTemplate(
        system=SYSTEM,
        instruction=INSTRUCTION
    )
)

PromptRegistry.register(
    "rhetoric_detection",
    PromptTemplate(
        system=RHETORIC_SYSTEM,
        instruction=RHETORIC_INSTRUCTION
    )
)

PromptRegistry.register(
    "seg",
    PromptTemplate(
        system=SEGMENTATION_SYSTEM,
        instruction=SEGMENTATION_INSTRUCTION
    )
)

PromptRegistry.register(
    "enetity_extraction",
    PromptTemplate(
        system=ENTITY_EXTRACTION_SYSTEM,
        instruction=ENTITY_EXTRACTION_INSTRUCTION
    )
)

PromptRegistry.register(
    'rewrite_with_wordbank',
    PromptTemplate(
        system=REWRITE_WITH_WORDBANK_SYSTEM,
        instruction=REWRITE_WITH_WORDBANK_INSTRUCTION
    )
)

PromptRegistry.register(
    "fluency_evaluation",
    PromptTemplate(
        system=FLUENCY_SYSTEM,
        instruction=FLUENCY_INSTRUCTION
    )
)

PromptRegistry.register(
    "similarity_evaluation",
    PromptTemplate(
        system=SIMILARITY_SYSTEM,
        instruction=SIMILARITY_INSTRUCTION
    )
)

PromptRegistry.register(
    "plain_rewrite",
    PromptTemplate(
        system=PLAIN_REWRITE_SYSTEM,
        instruction=PLAIN_REWRITE_INSTRUCTION
    )
)

PromptRegistry.register(
    "rewrite",
    PromptTemplate(
        system=REWRITE_SYSTEM,
        instruction=REWRITE_INSTRUCTION
    )
)