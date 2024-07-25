## Introduction
本版本在 https://github.com/microsoft/graphrag 的 Commit:61b5eea 基础上，增加了对qwen大模型的支持。

## Why do this
作为一个才开源不久的产品，`GraphRAG`尚不完善，例如：

* `GraphRAG`是基于`OpenAI`开发的，数据格式都采用`OpenAI`标准，但其他大模型并不一定这样。举例：在返回`embedding`数据时，`OpenAI`的返回值与`ollama`，以及`灵积dashscope`返回的值不同。

* 在`OpenAI`上跑`GraphRAG`时，费用很高，而且有些国家无法使用`OpenAI`的API，如果要在`llama3`等开源模型，或用`ollama`等本地大模型上运行的话，需要增加大量辅助工作。如果能在`GraphRAG`中，通过简单的配置，就适配大多数大模型，让更多的人方便地体验`GraphRAG`。

* 有些技术爱好者想在低配置笔记本上运行`GraphRAG`，但没有`OpenAI`的API，而使用非OpenAI大模型时，原生`GraphRAG`无法直接支持。

* 一些国产的大模型对中文的支持较好，在开发中文GPT应用时，国产大模型更准确，如果能配合`GraphRAG`知识图谱，将大幅加速人工智能在实际商业领域的应用。

## Quickstart
如果需要使用`qwen大模型`加与qwen配套的多语言文本向量模型的话，只要按下面正确设置`settings.yaml`即可，安装与运行同官方文档。

区别仅在于克隆本repo代码，即：
```
git clone https://github.com/jackygu2006/graphrag.git
```
其他参照官方文档，虽然使用pip也可以安装，但我偏好用poetry（作为一名nodejs开发者，python上的库管理工具poetry和yarn太像了😁）


下面是`settings.yaml`的推荐配置：
需要注意：
* `llm/model`：如果需要换成其他qwen的大模型，在这里调整，如：`qwen-turbo`, `qwen-long`等
* `llm/api_base`：如果使用qwen大模型，必须写：https://dashscope.aliyuncs.com/compatible-mode/v1
* `llm/max_tokens`：因为qwen大模型的限制，必须是2000，其他参数也用下面的即可，注意`top_p`必须在0到1之间，不能是0和1
* `embeddings/llm/type`: 必须写`qwen_embedding`，这是我新增的类别
* `embeddings/llm/model`: 我这里使用`text-embedding-v2`文本向量，具体参见[这里](https://help.aliyun.com/zh/dashscope/developer-reference/text-embedding-quick-start?spm=a2c4g.11186623.0.0.397e2fb4RD75YF)
* `embeddings/llm/max_tokens`: 填2048，因为该向量模型的限制。其他按下面配置即可。
* `global_search`：按下面配置，官方文档中没有`llm_top_p`，但因为qwen不支持top_p=1，所以需要打开这个选项，并设置一个比1小的值，否则无法执行推理。
* `local_search`: 同`global_search`。

```
encoding_model: cl100k_base
skip_workflows: []
llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat
  model: qwen-max
  model_supports_json: true # recommended if this is available for your model.
  api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
  max_tokens: 2000
  tokens_per_minute: 30000
  requests_per_minute: 30
  max_retries: 3
  max_retry_wait: 10
  sleep_on_rate_limit_recommendation: true
	suggests wait-times
  concurrent_requests: 1
  top_p: 0.85 # limit for qwen

parallelization:
  stagger: 0.3
  # num_threads: 50 # the number of threads to use for parallel processing

async_mode: threaded # or asyncio

embeddings:
  async_mode: threaded # or asyncio
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: qwen_embedding
    model: text-embedding-v2
    api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
    max_tokens: 2048
    tokens_per_minute: 30000 # set a leaky bucket throttle
    requests_per_minute: 30 # set a leaky bucket throttle
    max_retries: 3
    max_retry_wait: 10.0
    sleep_on_rate_limit_recommendation: true # whether to sleep when azure suggests wait-times
    concurrent_requests: 1 # the number of parallel inflight requests that may be made  
    top_p: 0.85

chunks:
  size: 300
  overlap: 100
  group_by_columns: [id] # by default, we don't allow chunks to cross documents
    
input:
  type: file # or blob
  file_type: text # or csv
  base_dir: "input"
  file_encoding: utf-8
  file_pattern: ".*\\.txt$"

cache:
  type: file # or blob
  base_dir: "cache"
  # connection_string: <azure_blob_storage_connection_string>
  # container_name: <azure_blob_storage_container_name>

storage:
  type: file # or blob
  base_dir: "output/${timestamp}/artifacts"
  # connection_string: <azure_blob_storage_connection_string>
  # container_name: <azure_blob_storage_container_name>

reporting:
  type: file # or console, blob
  base_dir: "output/${timestamp}/reports"
  # connection_string: <azure_blob_storage_connection_string>
  # container_name: <azure_blob_storage_container_name>

entity_extraction:
  ## llm: override the global llm settings for this task
  ## parallelization: override the global parallelization settings for this task
  ## async_mode: override the global async_mode settings for this task
  prompt: "prompts/entity_extraction.txt"
  entity_types: [organization,person,geo,event]
  max_gleanings: 0

summarize_descriptions:
  ## llm: override the global llm settings for this task
  ## parallelization: override the global parallelization settings for this task
  ## async_mode: override the global async_mode settings for this task
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500

claim_extraction:
  ## llm: override the global llm settings for this task
  ## parallelization: override the global parallelization settings for this task
  ## async_mode: override the global async_mode settings for this task
  # enabled: true
  prompt: "prompts/claim_extraction.txt"
  description: "Any claims or facts that could be relevant to information discovery."
  max_gleanings: 0

community_report:
  ## llm: override the global llm settings for this task
  ## parallelization: override the global parallelization settings for this task
  ## async_mode: override the global async_mode settings for this task
  prompt: "prompts/community_report.txt"
  max_length: 2000
  max_input_length: 8000

cluster_graph:
  max_cluster_size: 10

embed_graph:
  enabled: false # if true, will generate node2vec embeddings for nodes
  # num_walks: 10
  # walk_length: 40
  # window_size: 2
  # iterations: 3
  # random_seed: 597832

umap:
  enabled: false # if true, will generate UMAP embeddings for nodes

snapshots:
  graphml: false
  raw_entities: false
  top_level_nodes: false

local_search:
  text_unit_prop: 0.6
  community_prop: 0.1
  conversation_history_max_turns: 5
  top_k_mapped_entities: 10
  top_k_relationships: 10
  max_tokens: 2000
  llm_top_p: 0.85

global_search:
  max_tokens: 2000
  data_max_tokens: 2000
  map_max_tokens: 2000
  reduce_max_tokens: 2000
  concurrency: 1
  llm_top_p: 0.85

```

## TODO
大语言模型发展很快，版本的迭代更快，我希望能尽可能多的把GraphRAG在不同的大模型上运行。

* <input type="checkbox" name="option" value="option1" checked>  支持灵积平台上的文本向量embedding模型
* <input type="checkbox" name="option" value="option1">  支持灵积平台上非qwen系列大模型的支持，如：llama3，百川大模型，GLM大模型等等。
* <input type="checkbox" name="option" value="option1">  支持豆包平台上的大模型以及豆包的文本向量
* <input type="checkbox" name="option" value="option1">  支持ollama本地大模型


## 以下是GraphRAG官方文档 👇👇👇
# GraphRAG

👉 [Use the GraphRAG Accelerator solution](https://github.com/Azure-Samples/graphrag-accelerator) <br/>
👉 [Microsoft Research Blog Post](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)<br/>
👉 [Read the docs](https://microsoft.github.io/graphrag)<br/>
👉 [GraphRAG Arxiv](https://arxiv.org/pdf/2404.16130)

<div align="left">
  <a href="https://pypi.org/project/graphrag/">
    <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/graphrag">
  </a>
  <a href="https://pypi.org/project/graphrag/">
    <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/graphrag">
  </a>
  <a href="https://github.com/microsoft/graphrag/issues">
    <img alt="GitHub Issues" src="https://img.shields.io/github/issues/microsoft/graphrag">
  </a>
  <a href="https://github.com/microsoft/graphrag/discussions">
    <img alt="GitHub Discussions" src="https://img.shields.io/github/discussions/microsoft/graphrag">
  </a>
</div>

## Overview

The GraphRAG project is a data pipeline and transformation suite that is designed to extract meaningful, structured data from unstructured text using the power of LLMs.

To learn more about GraphRAG and how it can be used to enhance your LLMs ability to reason about your private data, please visit the <a href="https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/" target="_blank">Microsoft Research Blog Post.</a>

## Quickstart

To get started with the GraphRAG system we recommend trying the [Solution Accelerator](https://github.com/Azure-Samples/graphrag-accelerator) package. This provides a user-friendly end-to-end experience with Azure resources.

## Repository Guidance

This repository presents a methodology for using knowledge graph memory structures to enhance LLM outputs. Please note that the provided code serves as a demonstration and is not an officially supported Microsoft offering.

⚠️ *Warning: GraphRAG indexing can be an expensive operation, please read all of the documentation to understand the process and costs involved, and start small.*

## Diving Deeper

- To learn about our contribution guidelines, see [CONTRIBUTING.md](./CONTRIBUTING.md)
- To start developing _GraphRAG_, see [DEVELOPING.md](./DEVELOPING.md)
- Join the conversation and provide feedback in the [GitHub Discussions tab!](https://github.com/microsoft/graphrag/discussions)

## Prompt Tuning

Using _GraphRAG_ with your data out of the box may not yield the best possible results.
We strongly recommend to fine-tune your prompts following the [Prompt Tuning Guide](https://microsoft.github.io/graphrag/posts/prompt_tuning/overview/) in our documentation.

## Responsible AI FAQ

See [RAI_TRANSPARENCY.md](./RAI_TRANSPARENCY.md)

- [What is GraphRAG?](./RAI_TRANSPARENCY.md#what-is-graphrag)
- [What can GraphRAG do?](./RAI_TRANSPARENCY.md#what-can-graphrag-do)
- [What are GraphRAG’s intended use(s)?](./RAI_TRANSPARENCY.md#what-are-graphrags-intended-uses)
- [How was GraphRAG evaluated? What metrics are used to measure performance?](./RAI_TRANSPARENCY.md#how-was-graphrag-evaluated-what-metrics-are-used-to-measure-performance)
- [What are the limitations of GraphRAG? How can users minimize the impact of GraphRAG’s limitations when using the system?](./RAI_TRANSPARENCY.md#what-are-the-limitations-of-graphrag-how-can-users-minimize-the-impact-of-graphrags-limitations-when-using-the-system)
- [What operational factors and settings allow for effective and responsible use of GraphRAG?](./RAI_TRANSPARENCY.md#what-operational-factors-and-settings-allow-for-effective-and-responsible-use-of-graphrag)

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Privacy

[Microsoft Privacy Statement](https://privacy.microsoft.com/en-us/privacystatement)
