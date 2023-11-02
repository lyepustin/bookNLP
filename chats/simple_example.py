# File: LangChainchatOpenAI.py
# Author: Denys L
# Date: October 8, 2023
# Description:

import streamlit as st
from streamlit_extras.stateful_chat import chat, add_message

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

import sys
from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler


from dotenv import load_dotenv
import os
import time

import sys
import pathlib
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

import warnings
# Suppress UserWarning from ebooklib
warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib")
# warnings.filterwarnings("ignore", category=UserWarning, module="langchain")


class StreamingStdOutCallbackHandlerPersonal(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        st.session_state.full_response = st.session_state.full_response + token
        # st.message_placeholder.markdown(st.session_state.full_response + "▌")
        with st.chat_message("assistant", avatar="🤖"):
            st.write(st.session_state.full_response)
        sys.stdout.write(token)
        sys.stdout.flush()


def get_conversation_chain():
    handler = StreamingStdOutCallbackHandlerPersonal()
    llm = OpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.1, openai_api_key=os.getenv(
        "OPENAI_API_KEY"), streaming=True, callbacks=[handler],  verbose=True)
    return llm


def sent_message(conversation_chain, prompt):
    prompt_template = f"Answer this question: {prompt}"
    return conversation_chain(prompt_template)


def main():
    load_dotenv()
    st.title("ChatGPT-like storyteller")

    if "conversation" not in st.session_state:
        st.session_state.full_response = ""
        st.message_placeholder = st.empty()
        st.session_state.conversation = get_conversation_chain()

    with chat(key="my_chat"):
        if prompt := st.chat_input():
            add_message("user", prompt, avatar="🧑‍💻")
            message = sent_message(st.session_state.conversation, prompt)
            print(len(message))


if __name__ == '__main__':
    main()
