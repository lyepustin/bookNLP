# File: LangchainSummarizer.py
# Author: Denys L
# Date: October 8, 2023
# Description: 
#   Demonstrates a simple text summarization process using 
#   LangChain and OpenAI, specifically employing a map-reduce approach to 
#   summarize a portion of the text from the file "quijote.txt."


from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()
import os
import pathlib


   
def summarization_map_reduce(data_path):
    llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

    loader = TextLoader(data_path, encoding="utf-8")
    documents = loader.load()

    # Get your splitter ready
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

    # Split your docs into texts
    texts = text_splitter.split_documents(documents)

    # There is a lot of complexity hidden in this one line. I encourage you to check out the video above for more detail
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    chain.run(texts[:4])


def summarization_stuff(data_path):
    llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

    loader = TextLoader(data_path, encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=16000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Define prompt
    prompt_template = """Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM chain
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", openai_api_key=os.getenv("OPENAI_API_KEY"))
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="text", verbose=True
    )

    print(texts[0])
    print(stuff_chain.run([texts[0]]))




if __name__ == '__main__':
    path_parent = pathlib.Path(__file__).parent.parent.resolve()
    data_path = f"{path_parent}/data/quijote.txt"
    # summarization_map_reduce(data_path)
    summarization_stuff(data_path)
    # tutorial: https://python.langchain.com/docs/use_cases/summarization