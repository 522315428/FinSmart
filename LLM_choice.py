import os
import numpy as np
import pandas as pd
import re
import os
from tqdm import tqdm
import logging
import sys

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.indices.vector_store.base import VectorStoreIndex
import torch
# from transformers import pipeline
from typing import Optional, List, Mapping, Any
# from llama_index.core.service_context import ServiceContext
# from llama_index.readers.file import UnstructuredReader, PDFReader, SimpleDirectoryReader, SummaryIndex
# from llama_index.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.service_context import ServiceContext
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.llms.callbacks import llm_completion_callback
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from llama_index.core import PromptTemplate
from llama_index.core.service_context import set_global_service_context
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.indices.knowledge_graph.base import KnowledgeGraphIndex
from llama_index.core import Document
from llama_index.readers.file import UnstructuredReader, PDFReader
from llama_index.core import Settings
from pathlib import Path
from pyvis.network import Network
import IPython
from llama_index.core import load_index_from_storage

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# set context window size
context_window = 3900
# set number of output tokens
num_output = 256
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# store the pipeline/model outside of the LLM class to avoid memory issues
model_name = "internlm2-chat-20b"

tokenizer = AutoTokenizer.from_pretrained("/model/Weight/internlm2-chat-20b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/model/Weight/internlm2-chat-20b", device_map="auto",
                                             trust_remote_code=True, torch_dtype=torch.float16).eval()
model.generation_config = GenerationConfig.from_pretrained(f"/model/Weight/internlm2-chat-20b",
                                                           trust_remote_code=True)
embed_model = HuggingFaceEmbedding(model_name="/model/Weight/BAAI/bge-m3")


class LLM(CustomLLM):
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=context_window,
            num_output=num_output,
            model_name=model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        prompt_length = len(prompt)

        # only return newly generated tokens
        text, _ = model.chat(tokenizer, prompt, history=[])
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(
            self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError()


llm = LLM()

exl = pd.read_excel('/code/open_clip-main/IneternLM/金融大赛(公告)/2024-02-28-公告测评集（有选项）.xls', engine='xlrd')
print(exl[:5])

exl = exl[:1]

num = 0
mac = 0

re_prompt = """
可能涉及的领域有：'企业管理与决策分析', '财务与投资分析', '市场与法律合规', '风险与影响评估', '经营分析','人力资源管理'
请根据文本中的信息，告知身份与任务，领域，相关性（0-1）模板为：
实际领域：
身份：
任务：
企业管理与决策分析相关性：
财务与投资分析相关性：
市场与法律合规相关性：
风险与影响评估相关性：
文本分析与逻辑推理经营效率分析相关性：
人力资源管理相关性：
是否为计算题：
完成分析
请严格按照模板格式，无需新增内容
"""

new_summary_tmpl_str = (
    "提供的信息如下：\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Query: {query_str}\n"
    "请判断提供的信息与Query在内容与意思上是否完全一致，仅可使用提供的信息，不能使用先验知识。\n"
    "如果提供的信息为或者包含'No relationships found'，请在判断处回答‘不一致’；\n"
    "如果存在任何信息不完全匹配的情况，请在判断处回答‘不一致’；\n"
    "如果意思不符合的情况，请在判断处回答‘不一致’；\n"
    "如果不能完全确定，或者是“可能“的情况，请在判断处回答‘不一致’；\n"
    "如果间接推理得到的情况，请在判断处回答‘不一致’；"
    "如果两者在含义、内容或所表达的概念上完全吻合，请在判断处回答‘相同’"
    "否则回答‘相同’。\n"
    "请严格按照模板,模板如下所示：判断：\n，原因：\n，提供的信息为：\n，"
    "请回答: "
)
new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)

prompt_q = '''\n请仔细阅读公告内容，并抽取其中的关键事件。针对每个事件，请严格按照格式生成一个问句(无需作出回答)。
格式为:
问题：事件,可能导致的后果是？
问题：事件，可能导致的后果是？
 请输出：'''

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

username = "neo4j"
password = "12345678"
url_1 = "neo4j://10.6.44.224:1111"
url_2 = "neo4j://10.6.44.224:4444"
url_3 = "neo4j://10.6.44.224:5555"
url_4 = "neo4j://10.6.44.224:3333"
url_5 = "neo4j://10.6.44.224:2222"
url_6 = "neo4j://10.6.44.224:7687"

database = "neo4j"

graph_store1 = Neo4jGraphStore(username=username,password=password,url=url_1,database=database)
graph_store2 = Neo4jGraphStore(username=username,password=password,url=url_2,database=database)
graph_store3 = Neo4jGraphStore(username=username,password=password,url=url_3,database=database)
graph_store4 = Neo4jGraphStore(username=username,password=password,url=url_4,database=database)
graph_store5 = Neo4jGraphStore(username=username,password=password,url=url_5,database=database)
graph_store6 = Neo4jGraphStore(username=username,password='neo4j',url=url_6,database=database)


storage_context1 = StorageContext.from_defaults(persist_dir="/code/open_clip-main/IneternLM/book/战略管理", graph_store=graph_store1)
loaded_index_1 = load_index_from_storage(storage_context1)
neo4j_kg_engine_1 = loaded_index_1.as_query_engine(include_text=False, response_mode="tree_summarize")

storage_context2 = StorageContext.from_defaults(persist_dir="/code/open_clip-main/IneternLM/book/财务管理", graph_store=graph_store2)
loaded_index_2 = load_index_from_storage(storage_context2)
neo4j_kg_engine_2 = loaded_index_2.as_query_engine(include_text=False, response_mode="tree_summarize")

storage_context3 = StorageContext.from_defaults(persist_dir="/code/open_clip-main/IneternLM/book/法律", graph_store=graph_store3)
loaded_index_3 = load_index_from_storage(storage_context3)
neo4j_kg_engine_3 = loaded_index_3.as_query_engine(include_text=False, response_mode="tree_summarize")

storage_context4 = StorageContext.from_defaults(persist_dir="/code/open_clip-main/IneternLM/book/投资与风险", graph_store=graph_store4)
loaded_index_4 = load_index_from_storage(storage_context4)
neo4j_kg_engine_4 = loaded_index_4.as_query_engine(include_text=False, response_mode="tree_summarize")

storage_context5 = StorageContext.from_defaults(persist_dir="/code/open_clip-main/IneternLM/book/经营分析", graph_store=graph_store5)
loaded_index_5 = load_index_from_storage(storage_context5)
neo4j_kg_engine_5 = loaded_index_5.as_query_engine(include_text=False, response_mode="tree_summarize")

storage_context6 = StorageContext.from_defaults(persist_dir="/code/open_clip-main/IneternLM/book/人力", graph_store=graph_store6)
loaded_index_6 = load_index_from_storage(storage_context6)
neo4j_kg_engine_6 = loaded_index_6.as_query_engine(include_text=False, response_mode="tree_summarize")

for i in tqdm(range(len(exl))):
    text = exl['评测问题'][i].split('\n')
    n_t = '文本为：' + text[0] + '\n' + re_prompt
    result = llm.complete(n_t)

    new_summary_tmpl_str2 = (
            "公告信息如下：\n" + str(text[1]) +
            "\n提供的信息如下：\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "问题: {query_str}\n"
            "请根据公告信息、提供的信息针对问题给出尽可能详细的答案(仅输出答案即可)"
    )
    new_summary_tmpl2 = PromptTemplate(new_summary_tmpl_str2)

    try:
        space = re.search(r'实际领域：(.*?)\n', str(result)).group(1)
    except:
        space = '未知'
    try:
        job = re.search(r'身份：(.*?)\n', str(result)).group(1)
    except:
        job = '专家'
    try:
        task = re.search(r'任务：(.*?)\n', str(result)).group(1)
    except:
        task = '分析以下文本'
    try:
        s1 = re.search(r'企业管理与决策分析相关性：(.*?)\n', str(result)).group(1)
        s2 = re.search(r'财务与投资分析相关性：(.*?)\n', str(result)).group(1)
        s3 = re.search(r'市场与法律合规相关性：(.*?)\n', str(result)).group(1)
        s4 = re.search(r'风险与影响评估相关性：(.*?)\n', str(result)).group(1)
        s5 = re.search(r'文本分析与逻辑推理经营效率分析相关性：(.*?)\n', str(result)).group(1)
    except:
        s1 = s2 = s3 = s4 = s5 = 1

    spaces = ['企业管理与决策分析', '财务与投资分析', '市场与法律合规', '风险与影响评估', '经营分析', '人力资源管理']
    sn = [s1, s2, s3, s4, s5]
    results = '多方面公告理解为：'

    for sp in range(len(spaces)):
        # prompt = '请以' + str(spaces[sp]) + '的知识，并根据任务：' + task + '\n超级详细的详细解析公告,并超级详细说明带来的直接影响:' + text[1] + '\n仅输出解析即可'
        # print(prompt)
        query_sp = str(text[1]) + prompt_q
        ress2 = str(llm.complete(query_sp))
        query2 = ress2.split('问题')
        sp_result = ''
        for q in query2:
            if sp == 0:
                neo4j_kg_engine_1.update_prompts({"response_synthesizer:summary_template": new_summary_tmpl2})
                ress3 = neo4j_kg_engine_1.query(q)
            elif sp == 1:
                neo4j_kg_engine_2.update_prompts({"response_synthesizer:summary_template": new_summary_tmpl2})
                ress3 = neo4j_kg_engine_2.query(q)
            elif sp == 2:
                neo4j_kg_engine_3.update_prompts({"response_synthesizer:summary_template": new_summary_tmpl2})
                ress3 = neo4j_kg_engine_3.query(q)
            elif sp == 3:
                neo4j_kg_engine_4.update_prompts({"response_synthesizer:summary_template": new_summary_tmpl2})
                ress3 = neo4j_kg_engine_4.query(q)
            elif sp == 4:
                neo4j_kg_engine_5.update_prompts({"response_synthesizer:summary_template": new_summary_tmpl2})
                ress3 = neo4j_kg_engine_5.query(q)
            elif sp == 5:
                neo4j_kg_engine_6.update_prompts({"response_synthesizer:summary_template": new_summary_tmpl2})
                ress3 = neo4j_kg_engine_6.query(q)
            sp_result = sp_result + str(ress3) + '\n'

        results = results + '\n' + spaces[sp] + '方面分析意见为：' + '\n' + str(
            sp_result) + '\n'

    result_fin = '公告：' + text[1] + '\n' + results
    text_path = '/code/open_clip-main/IneternLM/choice_text/question' + str(i) + '.txt'
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(str(result_fin))

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        chunk_size=512
    )
    set_global_service_context(service_context)
    loader = UnstructuredReader()
    documents = loader.load_data(file=Path(text_path))
    graph_store = SimpleGraphStore()  # In-memory
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=50,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
        # include_embeddings=True, # Query with embeddings
    )
    g = index.get_networkx_graph()
    net = Network(notebook=True, cdn_resources="in_line", directed=True)
    net.from_nx(g)
    graph_path = '/code/open_clip-main/IneternLM/choice_graph/question' + str(i) + '.html'
    net.show(graph_path)
    IPython.display.HTML(filename=graph_path)
    basic_kg_engine = index.as_query_engine(include_text=False, response_mode="tree_summarize")
    # prompt_kg = basic_kg_engine.get_prompts()

    print('<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>')
    basic_kg_engine.update_prompts(
        {"response_synthesizer:summary_template": new_summary_tmpl}
    )

    print('<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>')
    choices = ''
    text_choice = ['A', 'B', 'C', 'D']
    for c in text_choice:
        xc = '选项' + str(c)
        query = "请问" + str(exl[xc][i]) + "吗?"
        # print(prompt3)
        # result_choice = llm.complete(prompt3)
        result_choice = basic_kg_engine.query(query)
        print(str(result_choice))
        print('--------------------------------')
        try:
            is_choice = re.search(r'判断：(.*?)\n', str(result_choice)).group(1)
        except:
            is_choice = '不相同'
        print(str(is_choice)[0:1])
        # if str(result_choice)[0] == '是':
        if str(is_choice)[0:1] == '相':
            choices = choices + str(c) + '、'
    choices = choices[:-1]
    print("\n选项为：" + str(choices))
    print('答案为：' + str(exl['答案'][i]))
    print('<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>')

    num = num + 1
    if choices == exl['答案'][i]:
        mac = mac + 1
        print(mac)
    exl['答案'][i] = choices

acc = mac / num
print('acc:', acc)

exl.to_excel('/code/open_clip-main/IneternLM/金融大赛(公告)/choice_Answer.xls', index=False)
