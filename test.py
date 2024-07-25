import dashscope
from http import HTTPStatus
import os
import json
from volcenginesdkarkruntime import Ark
from openai import OpenAI
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

def ollama_embedding_response_to_openai(json_data: dict) -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse(
			  data=json_data["data"],
        model=json_data["model"],
        object="list",
        usage=Usage(
					  prompt_tokens=0,
					  total_tokens=0
				)
		)

def ollama_embed_with_str():
    ## 返回格式：(embedding=[0.018847845, 0.047919013, -0.1637811, ...], index=0, object='embedding')], model='nomic-embed-text', object='list', usage=None)
    client = OpenAI(
				base_url="http://127.0.0.1:11434/v1",
   			api_key="ollama"
		)
    resp = client.embeddings.create(
        model="nomic-embed-text", # 这是endpoint id
        input=["花椰菜又称菜花、花菜，是一种常见的蔬菜。"]
		)
    return(resp)

def llama_cpp_embed_with_str():
    ## load nomic-embed-text from local llama.cpp
    ## This is OAI compatible
    client = OpenAI(
				base_url="http://127.0.0.1:8080/v1",
				api_key="llama.cpp"
		)
    resp = client.embeddings.create(
				model="anything-can-be",
				input=["Hello, how are you"]
		)
    print(resp)
    # print(json.dumps(resp, indent=2))
    
def qwen_embed_with_str() -> dict:
    # 阿里dashscope
    dashscope.api_key="sk-0235ca4a71bd4815a691ee0697853006"
    resp = dashscope.TextEmbedding.call(
        model=dashscope.TextEmbedding.Models.text_embedding_v2,
        input='衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买')
    if resp.status_code == HTTPStatus.OK:
	      # 返回值可能重复，去重处理
        json_data = str(resp).split('}{')[0]
        return json.loads(json_data)
    else:
        return None

def doubao_embed_with_str():
    ## 抖音火山方舟大模型
    base_url = "https://ark.cn-beijing.volces.com/api/v3/embeddings"
    client = Ark(
      	base_url=base_url,
      	api_key="3b7d2cc5-3d9d-46ca-9beb-4120445e7164"
    )

    print("----- embeddings request -----")
    resp = client.embeddings.create(
        model="ep-20240725104921-zc4kx", # 这是endpoint id
        input=["花椰菜又称菜花、花菜，是一种常见的蔬菜。"]
		)
    print(resp)

def from_ali_to_openai_embedding():
    print("from_ali_to_openai_embedding")

def from_ollama_to_openai_embedding():
    print("from_ollama_to_openai_embedding")
    
def from_doubao_to_openai_embedding():
    print("from_doubao_to_openai_embedding")

if __name__ == '__main__':
    # llama_cpp_embed_with_str()
    
    json_data = qwen_embed_with_str()
    print(qwen_embedding_response_to_openai(json_data, "text-embedding-v2"))
    
    # json_data = ollama_embed_with_str()
    # print(ollama_embedding_response_to_openai(json.dumps(json_data, indent=2)))
