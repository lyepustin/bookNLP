import os
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI


load_dotenv()
st.title("ChatGPT-like storyteller")

openai_api_key = os.getenv("OPENAI_API_KEY")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# if prompt := st.chat_input("What is up?"):
prompt = "Quiero que te inventes un cuento largo sobre sobre un pato con un sombrero de playa, gafas negras, cadena de oro y un flotador inchanble"

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in OpenAI(temperature=0.7, openai_api_key=openai_api_key)(prompt):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})


# Define the scroll operation as a function and pass in something unique for each
# page load that it needs to re-evaluate where "bottom" is
# js = f"""
# <script>
#     function scroll(dummy_var_to_force_repeat_execution){{
#         var textAreas = parent.document.querySelectorAll('section.main');
#         for (let index = 0; index < textAreas.length; index++) {{
#             textAreas[index].style.color = 'blue'
#             textAreas[index].scrollTop = textAreas[index].scrollHeight;
#         }}
#     }}
#     scroll({len(st.session_state.messages)})
# </script>
# """

# st.components.v1.html(js)