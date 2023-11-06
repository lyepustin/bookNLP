# File: LangChainchatOpenAI.py
# Author: Denys L
# Date: October 8, 2023
# Description:


import os
import sys
import hashlib
from typing import Any
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from fundamentals.langchain_utils import StuffSummarizerByChapter


class StreamingStdOutCallbackHandlerPersonal(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        st.session_state.full_response = st.session_state.full_response + token
        st.session_state.placeholder.markdown(
            st.session_state.full_response + "▌")
        sys.stdout.write(token)
        sys.stdout.flush()


def process_book(uploaded_file):
    temp_file_path = f'.trash/{uploaded_file.name}'
    with open(temp_file_path, 'wb') as file:
        file.write(uploaded_file.read())

    st.session_state.full_response = ""
    st.session_state.handler_ia_message = st.chat_message(
        "assistant", avatar="🤖")
    st.session_state.placeholder = st.session_state.handler_ia_message.empty()
    # magic
    st.session_state.llm.summarize(temp_file_path)
    # print output
    st.session_state.placeholder.markdown(st.session_state.full_response)
    st.session_state.messages.append(
        {"role": "assistant", "content": st.session_state.full_response, "avatar": "🤖"})
    st.session_state.full_response = ""

    # remove temp file
    os.remove(temp_file_path)


def main():
    load_dotenv()
    st.title("Storyteller")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.handler = StreamingStdOutCallbackHandlerPersonal()
        st.session_state.llm = StuffSummarizerByChapter(
            st.session_state.handler)

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])

    st.sidebar.subheader("Your books")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your Books here and click on 'Process' to start the story", accept_multiple_files=False)
    if st.sidebar.button("Process"):
        with st.spinner("Processing"):
            process_book(uploaded_file)


if __name__ == '__main__':
    main()
