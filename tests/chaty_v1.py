import time
import sys, os
import pathlib
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate 
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.mapreduce import MapReduceChain, ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter


from dotenv import load_dotenv
load_dotenv()

class TextProcessor:
    def __init__(self):
        self.split_docs = None
        self.raw_documents = {}
        self.documents = []
        self.chunk_size = int(os.getenv("TEXT_SPLITTER_CHUNK_SIZE"))
        self.chunk_overlap = int(os.getenv("TEXT_SPLITTER_CHUNK_OVERLAP"))
        self.min_content = int(os.getenv("BOOK_MIN_CONTENT_PER_CHAPTER"))
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

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
                            'chapter' : chapter,
                            'document_create_time' : time.time()
                        }))

    def process_book(self):
        chapter = "Cover"
        for item in epub.read_epub(self.data_path).get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_body_content().decode('utf-8'), "html.parser")
                if soup.find("h2"):
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


class StreamingStdOutCallbackHandlerPersonal(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # yield token + " "
        sys.stdout.write(token)
        sys.stdout.flush()

    
def main():
    load_dotenv()

    data_path = os.getenv("BOOK_PATH")
    text_processor = TextProcessor()
    text_processor.process_file(data_path)
    for chapter, docs in text_processor.grouped_docs.items():
        prompt_template = """Write a concise summary of the following chapter:"%chapter%". Text:"{text}" CONCISE SUMMARY:In the %chapter% chapter..."""
        prompt_template = """Escriba un resumen completo. Texto:"{text}" RESUMEN COMPLETO:En el capítulo %chapter%, ..."""
        prompt_template = prompt_template.replace("%chapter%", chapter)
        prompt = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.1, openai_api_key=os.getenv("OPENAI_API_KEY"), verbose=True)
        llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text", verbose=True, callbacks=[StreamingStdOutCallbackHandlerPersonal(token)])
        stuff_chain.run(docs)
        sys.exit()
    


if __name__ == '__main__':
    main()
