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


   
def summarization_map_reduce_v0(data_path):
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


def summarization_map_reduce(data_path):
    from langchain.chains.mapreduce import MapReduceChain
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain

    token_max = 16000
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", openai_api_key=os.getenv("OPENAI_API_KEY"), verbose=True)

    # Map
    map_template = """The following is a set of documents
    {docs}
    Based on this list of docs, please identify the main themes 
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt, verbose=True)

    # Reduce
    reduce_template = """The following is set of summaries:
    {doc_summaries}
    Take these and distill it into a final, consolidated summary of the main themes. 
    Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
        
    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt, verbose=True)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries", verbose=True
    )

    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        # token_max=4000,
        token_max=token_max,
        verbose=True
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
        verbose=True
    )

    loader = TextLoader(data_path, encoding="utf-8")
    documents = loader.load()
    # Get your splitter ready
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=token_max, chunk_overlap=50)
    # Split your docs into texts
    split_docs = text_splitter.split_documents(documents)

    print(map_reduce_chain.run(split_docs[:3]))



if __name__ == '__main__':
    path_parent = pathlib.Path(__file__).parent.parent.resolve()
    data_path = f"{path_parent}/data/quijote.txt"
    summarization_map_reduce(data_path)
    # summarization_stuff(data_path)
    # tutorial: https://python.langchain.com/docs/use_cases/summarization