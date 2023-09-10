import streamlit as st
from streamlit_extras.stateful_chat import chat, add_message

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings

from dotenv import load_dotenv
import qdrant_client
import os
import time


def get_qdrant_client():
    return qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_HOST"),
        port=os.getenv("QDRANT_PORT"),
        api_key=os.getenv("QDRANT_API_KEY"),
        https=True)


def get_vector_store():
    client = get_qdrant_client()

    embeddings = OpenAIEmbeddings()

    return Qdrant(
        client=client, 
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
        embeddings=embeddings,
    )


def get_conversation_chain():
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=get_vector_store().as_retriever(),
        memory=memory
    )
    return conversation_chain


def main():
    load_dotenv()
    st.set_page_config(
        page_title="Chat with multiple Book's",
        page_icon=":books:",
        layout="centered",
    )   
    st.header("Ask your remote database 💬")

    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain()

    with chat(key="my_chat"):
        if prompt := st.chat_input():
            add_message("user", prompt, avatar="🧑‍💻")
            response = st.session_state.conversation({'question': prompt})
            def stream_echo():
                for word in response['answer'].split():
                    yield word + " "
                    time.sleep(0.10)
            add_message("assistant", stream_echo, avatar="🤖")


if __name__ == '__main__':
    main()
