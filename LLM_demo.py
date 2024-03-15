import os
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import os
from tqdm import tqdm

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

# llm = OpenAILike(
#     # model="internlm2-chat-20b",
#     # model = "Qwen-14B-Chat",
#     model = 'kagentlms_qwen_7b_mat',
#     api_key ="EMPTY",
#     # api_base="http://10.6.80.75:8001/v1",
#     api_base = "http://10.6.59.88:8001/v1",
#     is_chat_model=True
#     )

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
from llama_index.core.llms.callbacks import llm_completion_callback
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# set context window size
context_window = 3900
# set number of output tokens
num_output = 256

# store the pipeline/model outside of the LLM class to avoid memory issues
model_name = "internlm2-chat-20b"

tokenizer = AutoTokenizer.from_pretrained("/model/Weight/internlm2-chat-20b", trust_remote_code=True)
# model = AutoModel.from_pretrained("Qwen-7B-Chat", trust_remote_code=True, device='cuda')
model = AutoModelForCausalLM.from_pretrained("/model/Weight/internlm2-chat-20b", device_map="auto",
                                                 trust_remote_code=True, torch_dtype=torch.float16).eval()

    # model = model.eval()
model.generation_config = GenerationConfig.from_pretrained(f"/model/Weight/internlm2-chat-20b",
                                                               trust_remote_code=True)


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


def main():
    llm = LLM()
    exl = pd.read_excel('/code/open_clip-main/IneternLM/choice.xls', engine='xlrd')
    print(exl[:5])

    num = 0
    mac = 0

    # print(exl['评测问题'][0])

    text_choice = """
    请以不定项选择的方式给出答案，下面是回答模板：
    答案为：B、C
    下面开始作答，严格按照模板格式
    """
    print(len(exl))
    for i in tqdm(range(len(exl))):
        question = "评测问题：" + str(exl['评测问题'][i]) + '\n' + '选项为' + 'A:' + str(exl['选项A'][i]) + '\n' + \
                   '选项为' + 'B:' + str(exl['选项B'][i]) + '\n' + '选项为' + 'C:' + str(exl['选项C'][i]) + '\n' + \
                   '选项为' + 'D:' + str(exl['选项D'][i]) + '\n' + text_choice
        # print(question)
        # print('_____________________________________________')

        # LlamaIndex
        # result = llm.complete('hello how are you')
        # result = llm.complete(text + '/n' + exl['评测问题'][0])
        result = llm.complete(question)
        # print(result)
        try:
            matches = re.findall(r'答案为：([A-Z](?:、[A-Z])*)', str(result))[0]
        # matches = re.search(r'答案为：(.*)', str(result)).group(1)
        # print("===========")
        except:
            matches = []
        # print("===========")
        # print(matches)
        print(matches)
        print('||||||||||||')
        print(exl['答案'][i])
        print('===========')
        num = num + 1
        if matches == exl['答案'][i]:
            mac = mac + 1
            print(mac)
        acc = mac / num

    print('=======================')
    print('acc:', acc)
    # embed_model = HuggingFaceEmbedding(model_name="/model/Weight/BAAI/bge-small-en-v1.5")


if __name__ == "__main__":
    main()
