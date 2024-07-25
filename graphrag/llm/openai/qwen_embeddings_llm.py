# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The EmbeddingsLLM class."""
import json
from typing_extensions import Unpack

from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    EmbeddingInput,
    EmbeddingOutput,
    LLMInput,
)

from .openai_configuration import OpenAIConfiguration
from .types import OpenAIClientTypes
import dashscope
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Embedding:
    object: str
    embedding: List[float]
    index: int

@dataclass
class Usage:
    prompt_tokens: int
    total_tokens: int

@dataclass
class CreateEmbeddingResponse:
    object: str
    data: List[Embedding]
    usage: Usage
    model: str

class QwenEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    """A text-embedding generator LLM."""

    # _client: OpenAIClientTypes
    # _configuration: OpenAIConfiguration

    def __init__(self, configuration: OpenAIConfiguration):
        # self.configuration = configuration
        dashscope.api_key = configuration.api_key

    def qwen_embedding_response_to_openai(json_data: dict, model: str) -> CreateEmbeddingResponse:
        return CreateEmbeddingResponse(
						data=[
								Embedding(
										embedding=emb["embedding"], 
										index=emb["text_index"], 
										object="embedding"
								) for emb in json_data["output"]["embeddings"]
						],
						model=model,
						object="list",
						usage=Usage(
								prompt_tokens=json_data["usage"]["total_tokens"],
								total_tokens=json_data["usage"]["total_tokens"]
						),
				)

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
      
        resp = dashscope.TextEmbedding.call(
						model=dashscope.TextEmbedding.Models.text_embedding_v2,
						input=input,
				)
        json_data = dict(json.loads(str(resp).split('}{')[0]))
        
        embedding = CreateEmbeddingResponse(
						data=[
								Embedding(
										embedding=emb["embedding"], 
										index=emb["text_index"], 
										object="embedding"
								) for emb in json_data["output"]["embeddings"]
						],
						model="qwen_embedding",
						object="list",
						usage=Usage(
								prompt_tokens=json_data["usage"]["total_tokens"],
								total_tokens=json_data["usage"]["total_tokens"]
						),
				)
        
        # args = {
        #     "model": self.configuration.model,
        #     **(kwargs.get("model_parameters") or {}),
        # }
        # embedding = await self.client.embeddings.create(
        #     input=input,
        #     **args,
        # )
        return [d.embedding for d in embedding.data]
