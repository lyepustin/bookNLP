# File: langchain_summarizer.py
# Author: Denys L
# Date: October 8, 2023
# Description: Demonstrates a simple text summarization process using LangChain and OpenAI.

import sys
from dotenv import load_dotenv
import os
import pathlib
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import time

from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.mapreduce import MapReduceChain, ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()


class TextProcessor:
    def __init__(self):
        self.split_docs = None
        self.raw_documents = {}
        self.documents = []
        self.chunk_size = int(os.getenv("TEXT_SPLITTER_CHUNK_SIZE"))
        self.chunk_overlap = int(os.getenv("TEXT_SPLITTER_CHUNK_OVERLAP"))
        self.min_content = int(os.getenv("BOOK_MIN_CONTENT_PER_CHAPTER"))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def process_text(self):
        loader = TextLoader(self.data_path, encoding="utf-8")
        self.split_docs = self.text_splitter.split_documents(loader.load())

    def process_raw_docs(self):
        for chapter, texts_list in self.raw_documents.items():
            texts = " ".join(texts_list)
            if len(texts) > self.min_content:
                self.documents.append(
                    Document(
                        page_content=texts,
                        metadata={
                            'source': self.data_path,
                            'chapter': chapter,
                            'document_create_time': time.time()
                        }))

    def process_book(self):
        import warnings
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="ebooklib.*")
        for item in epub.read_epub(self.data_path, {"ignore_ncx": False}).get_items():
            chapter = "Unknown"
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(
                    item.get_body_content().decode('utf-8'), "html.parser")
                if soup.find("h1"):
                    chapter = soup.find("h1").get_text()     
                elif soup.find("h2"):
                    chapter = soup.find("h2").get_text()
                paragraphs = soup.find_all("p")
                for paragraph in paragraphs:
                    if self.raw_documents.get(chapter):
                        texts = self.raw_documents.get(chapter)
                        texts.append(paragraph.get_text())
                        self.raw_documents.update({
                            chapter: texts
                        })
                    else:
                        self.raw_documents.update({
                            chapter: [paragraph.get_text()]
                        })

        self.process_raw_docs()
        self.split_docs = self.text_splitter.split_documents(self.documents)
        self.grouped_docs = {}
        for doc in self.split_docs:
            grouped_doc = self.grouped_docs.get(doc.metadata.get('chapter'))
            if grouped_doc:
                grouped_doc.append(doc)
                self.grouped_docs.update({
                    doc.metadata.get('chapter'): grouped_doc
                })
            else:
                self.grouped_docs.update({
                    doc.metadata.get('chapter'): [doc]
                })

    def process_file(self, data_path):
        self.data_path = data_path
        _, file_extension = os.path.splitext(self.data_path)
        if file_extension.lower() == '.txt':
            self.process_text()
        elif file_extension.lower() == '.epub':
            self.process_book()
        else:
            print("Unsupported file format")
            sys.exit()


class StuffSummarizerByChapter:
    """Chain that combines documents by stuffing into context.

    This chain takes a list of documents and first combines them into a single string.
    It does this by formatting each document into a string with the `document_prompt`
    and then joining them together with `document_separator`. It then adds that new
    string to the inputs with the variable name set by `document_variable_name`.
    Those inputs are then passed to the `llm_chain`. """

    def __init__(self, callbackHandler):
        self.text_processor = TextProcessor()
        self.callbackHandler = callbackHandler

    def summarize(self, data_path):
        self.text_processor.process_file(data_path)
        for chapter, docs in self.text_processor.grouped_docs.items():
            prompt_template = """Write a concise summary of the following chapter:"%chapter%". Text:"{text}" CONCISE SUMMARY:In the %chapter% chapter..."""
            prompt_template = """Escriba un resumen completo. Texto:"{text}"  RESUMEN COMPLETO:En el capítulo %chapter%, ..."""
            prompt_template = """Escriba un resumen corto. Texto:"{text}"  RESUMEN CORTO:En el capítulo %chapter%, ..."""
            prompt_template = prompt_template.replace("%chapter%", chapter)
            prompt = PromptTemplate.from_template(prompt_template)
            llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo-16k", openai_api_key=os.getenv(
                "OPENAI_API_KEY"), streaming=True, callbacks=[self.callbackHandler], verbose=True, request_timeout=20)
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            stuff_chain = StuffDocumentsChain(
                llm_chain=llm_chain, document_variable_name="text", verbose=True)

            print(f"Staring processing: {chapter} with {len(docs)} docs.")
            stuff_chain.run(docs)
            self.callbackHandler.on_llm_new_token("  \n   \n")
            print(f"Treatment completed: {chapter} with {len(docs)} docs.")
