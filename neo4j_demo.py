import logging
import sys
import warnings
from llama_index.core.service_context import ServiceContext
from llama_index.core.indices.vector_store.base import VectorStoreIndex
import os
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from LLM_demo import LLM

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.service_context import set_global_service_context
from llama_index.core import Settings
# from llama_hub.file.unstructured.base import UnstructuredReader
from llama_index.readers.file import UnstructuredReader,PDFReader
from pathlib import Path
from llama_index.core.schema import Document
from llama_index.core.node_parser import SimpleNodeParser


from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.indices.knowledge_graph.base import KnowledgeGraphIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore
# from llama_index.graph_stores.nebula import NebulaGraphStore
from nebula import NebulaGraphStore
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core.query_engine import KnowledgeGraphQueryEngine

from pyvis.network import Network
import IPython
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Supress warnings
# warnings.filterwarnings("ignore")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# llm = OpenAILike(
#     # model="internlm2-chat-20b",
#     model = "Qwen-14B-Chat",
#     # model = 'kagentlms_qwen_7b_mat',
#     api_key ="EMPTY",
#     api_base="http://10.6.80.75:8001/v1",
#     # api_base = "http://10.6.59.88:8001/v1",
#     is_chat_model=True
#     )
llm = LLM()

print('{{{{{{{{{}}}}}}}}}}}}}}}')
result = llm.complete("你是谁？你可以回答金融问题吗")
print(result)
print('+++++++++++++++++++++++++++++++++++++++++++')

embed_model = HuggingFaceEmbedding(model_name="/model/Weight/BAAI/bge-m3")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    chunk_size=512
)

print('设置context----------------------------------')

set_global_service_context(service_context)
loader = UnstructuredReader()
# loader = PDFReader()
# paths = ['/code/open_clip-main/IneternLM/A_Lightweight_and_Multi_Branch_Module_in_Facial_Semantic_Segmentation_Feature_Extraction.pdf']
paths = ['/code/open_clip-main/IneternLM/book/人力/1_70.docx', '/code/open_clip-main/IneternLM/book/人力/71_140.docx',
         '/code/open_clip-main/IneternLM/book/人力/141_210.docx', '/code/open_clip-main/IneternLM/book/人力/211_280.docx',
         '/code/open_clip-main/IneternLM/book/人力/281_350.docx', '/code/open_clip-main/IneternLM/book/人力/351_420.docx',
         '/code/open_clip-main/IneternLM/book/人力/421_490.docx', '/code/open_clip-main/IneternLM/book/人力/491_560.docx',
         '/code/open_clip-main/IneternLM/book/人力/561_630.docx', '/code/open_clip-main/IneternLM/book/人力/631_700.docx',
         '/code/open_clip-main/IneternLM/book/人力/701_718.docx']

documents = []
for i in paths:
    document = loader.load_data(
            file=Path(i)
        )
    documents += document
# print(document.size)
# file=Path(f"/code/open_clip-main/IneternLM/A_Lightweight_and_Multi_Branch_Module_in_Facial_Semantic_Segmentation_Feature_Extraction.pdf")

print('设置环境=========================================')

# vector_index = VectorStoreIndex.from_documents(documents, service_context=service_context)
# baseline_engine = vector_index.as_query_engine(llm=llm)

# print("baseline+++++++++++++++++++++++++++++++++++++++")

query = '什么是人力资源管理 ?'  # Should be $6.08 billion

# print('query___________________________________________')
# ress = baseline_engine.query(query)

# print('ress___________________________________________')
#
# print(f'Question: {query}:\n\nAnswer: {ress.response}')

print('neo4j_____________')

import time
t0 = time.time()
# from llama_index.graph_stores import Neo4jGraphStore

username = "neo4j"
password = "neo4j"
url = "neo4j://10.6.44.224:7687"
database = "neo4j"

# 7687 人力资源        《人力资源管理》
# 5555 市场与法律合规   《》
# 4444 财务与投资分析   《会计财务管理》 《投资学》
# 3333 风险与影响评估   《风险分析与管理指南》
# 6666 企业管理与决策分析 《战略管理》
# 2222 经营效率分析     《经营分析与评价》



graph_store = Neo4jGraphStore(
    username=username,
    password=password,
    url=url,
    database=database,
)

print('成功构建graph store——————————————————————————————————————————————————')

storage_context = StorageContext.from_defaults(graph_store=graph_store)

print('开始构建Index——————————————————————————————————————————————————')

# NOTE: can take a while!
index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=9999999,
    service_context=service_context,
    show_progress=True,
)

print('开始kg Engine——————————————————————————————————————————————————')

neo4j_kg_engine = index.as_query_engine(include_text=False, response_mode="tree_summarize")
# basic_kg_engine = index.as_query_engine(include_text=False, response_mode="tree_summarize", embedding_mode="hybrid", similarity_top_k=5,) # Query with embeddings



t1 = time.time()
print(t1 - t0, 's')

index.storage_context.persist(persist_dir="/code/open_clip-main/IneternLM/book/人力")

g = index.get_networkx_graph()
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)
net.show('/code/open_clip-main/IneternLM/book/人力/renli.html')

import IPython
IPython.display.HTML(filename='/code/open_clip-main/IneternLM/book/人力/renli.html')

# query = 'what is the gross cost of operating lease vehicles as of december 31, 2022?' # Should be $6.08 billion
ress = neo4j_kg_engine.query(query)

print(f'Question: {query}:\n\nAnswer: {ress.response}')


