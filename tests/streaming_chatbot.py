# File: LangChainchatOpenAI.py
# Author: Denys L
# Date: October 8, 2023
# Description:

import streamlit as st
from streamlit_extras.stateful_chat import chat, add_message
from streamlit_extras.streaming_write import write as streamlit_write


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


class StreamingStdOutCallbackHandlerPersonal(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        st.session_state.full_response = st.session_state.full_response + token
        st.session_state.placeholder.markdown(
            st.session_state.full_response + "▌")
        sys.stdout.write(token)
        sys.stdout.flush()


def handle_question(prompt):
    st.session_state.full_response = ""
    st.session_state.handler_ia_message = st.chat_message(
        "assistant", avatar="🤖")
    st.session_state.placeholder = st.session_state.handler_ia_message.empty()
    response = st.session_state.llm(prompt)
    st.session_state.placeholder.markdown(st.session_state.full_response)
    return response


def main():
    load_dotenv()
    st.title("ChatGPT-like storyteller")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.handler = StreamingStdOutCallbackHandlerPersonal()
        st.session_state.llm = OpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.1, openai_api_key=os.getenv(
            "OPENAI_API_KEY"), streaming=True, callbacks=[st.session_state.handler],  verbose=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "avatar": "🧑‍💻"})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        response = handle_question(prompt)
        st.session_state.messages.append(
            {"role": "assistant", "content": response, "avatar": "🤖"})
        print(len(response))


if __name__ == '__main__':
    main()
