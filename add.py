from langchain.llms import OpenAI
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from ebooklib import epub
from bs4 import BeautifulSoup
import qdrant_client
import ebooklib
import logging
import os
import sys

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from qdrant_client.http.models.models import Filter
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import Distance, VectorParams

from chat import get_qdrant_client, get_vector_store
# def get_qdrant_client():
#     return qdrant_client.QdrantClient(
#         url=os.getenv("QDRANT_HOST"),
#         port=os.getenv("QDRANT_PORT"),
#         api_key=os.getenv("QDRANT_API_KEY"),
#         https=True)


# def get_vector_store():
#     client = get_qdrant_client()

#     embeddings = OpenAIEmbeddings()

#     return Qdrant(
#         client=client, 
#         collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
#         embeddings=embeddings,
#     )


def recreate_qdrant_collection(collection_name, size):
    
    client = get_qdrant_client()
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )
        logging.info(f"'{collection_name}' collection re-created.")
    except Exception as e:
        logging.error(f"on create collection '{collection_name}'. " + str(e).replace('\n',' '))


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator=str(os.getenv("TEXT_SPLITTER_SEPARATOR")),
        chunk_size=int(os.getenv("TEXT_SPLITTER_CHUNK_SIZE")),
        chunk_overlap=int(os.getenv("TEXT_SPLITTER_CHUNK_OVERLAP")),
        length_function=len
    )
    chunks = text_splitter.split_text(str(text))
    return chunks


def add_some_text():
    recreate_qdrant_collection(
        os.getenv("QDRANT_COLLECTION_NAME"), os.getenv("QDRANT_COLLECTION_SIZE"))
    
    text = os.getenv("TEXT_SAMPLE")
    text_chunks = get_text_chunks(text)
    vector_store = get_vector_store()
    ids = vector_store.add_texts(text_chunks)
    
    if len(ids) > 1:
        logging.info(
            f"partial content of book '{os.getenv('BOOK_NAME')}' successfully added " +
            f"to the '{os.getenv('VECTOR_DATABASE')}' vector database.")


def main():
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    
    options = {
        # '--full': add_book,
        '--some': add_some_text,
        # "--test": test_llm
    }
    flag = sys.argv[1]
    if flag in options:
        options.get(flag)()
    else:
        logging.error(
            f"Invalid flag. Please use: " + 
            " ".join([key for key, value in options.items()]))


if __name__ == '__main__':
    main()
