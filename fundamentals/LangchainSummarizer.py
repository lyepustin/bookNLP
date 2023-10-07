from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate

from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


from dotenv import load_dotenv
load_dotenv()
import os
import pathlib
import textwrap

   
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



if __name__ == '__main__':
    path_parent = pathlib.Path(__file__).parent.parent.resolve()
    data_path = f"{path_parent}/data/quijote.txt"
    summarization_map_reduce(data_path)


# import os

# separator = os.path.sep