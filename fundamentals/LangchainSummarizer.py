# File: langchain_summarizer.py
# Author: Denys L
# Date: October 8, 2023
# Description: Demonstrates a simple text summarization process using LangChain and OpenAI.

from dotenv import load_dotenv
import os
import pathlib

from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.mapreduce import MapReduceChain, ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

TOKEN_MAX = 16000

class TextProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.documents = None
        self.split_docs = None

    def process_text(self):
        loader = TextLoader(self.data_path, encoding="utf-8")
        self.documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=TOKEN_MAX, chunk_overlap=50)
        self.split_docs = text_splitter.split_documents(self.documents)

class RefineSummarizer:
    def __init__(self, data_path):
        self.text_processor = TextProcessor(data_path)

    def summarize(self):
        self.text_processor.process_text()
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", openai_api_key=os.getenv("OPENAI_API_KEY"))
        chain = load_summarize_chain(llm, chain_type="refine", verbose=True)
        print(chain.run(self.text_processor.split_docs[:3]))

class StuffSummarizer:
    def __init__(self, data_path):
        self.text_processor = TextProcessor(data_path)

    def summarize(self):
        self.text_processor.process_text()
        prompt_template = """Write a concise summary of the following: "{text}" CONCISE SUMMARY:"""
        prompt = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", openai_api_key=os.getenv("OPENAI_API_KEY"))
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text", verbose=True)
        print(stuff_chain.run(self.text_processor.split_docs[:3]))

class MapReduceSummarizer:
    def __init__(self, data_path):
        self.text_processor = TextProcessor(data_path)

    def summarize(self):
        self.text_processor.process_text()
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", openai_api_key=os.getenv("OPENAI_API_KEY"), verbose=True)
        map_template = """The following is a set of documents {docs} Based on this list of docs, please identify the main themes Helpful Answer:"""
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=llm, prompt=map_prompt, verbose=True)
        reduce_template = """The following is set of summaries: {doc_summaries} Take these and distill it into a final, consolidated summary of the main themes. Helpful Answer:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt, verbose=True)
        combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="doc_summaries", verbose=True)
        reduce_documents_chain = ReduceDocumentsChain(combine_documents_chain=combine_documents_chain,
                                                      collapse_documents_chain=combine_documents_chain,
                                                      token_max=TOKEN_MAX, verbose=True)
        map_reduce_chain = MapReduceDocumentsChain(llm_chain=map_chain, reduce_documents_chain=reduce_documents_chain,
                                                   document_variable_name="docs", return_intermediate_steps=False, verbose=True)
        print(map_reduce_chain.run(self.text_processor.split_docs[:3]))

if __name__ == '__main__':
    path_parent = pathlib.Path(__file__).parent.parent.resolve()
    data_path = f"{path_parent}/data/quijote.txt"
    
    # RefineSummarizer(data_path).summarize()
    # StuffSummarizer(data_path).summarize()
    MapReduceSummarizer(data_path).summarize()
    # tutorial: https://python.langchain.com/docs/use_cases/summarization
