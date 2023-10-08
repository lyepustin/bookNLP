# File: LangChainchatOpenAI.py
# Author: Denys L
# Date: October 8, 2023
# Description: 
#   This snippet demonstrates how to use the LangChain library to create a chat 
#   interaction with an OpenAI language model. The script sets up a conversation 
#   with a system message, human input, AI-generated response, and additional 
#   human input, showcasing the interaction flow with the language model.



from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()
import os

# If the temperature is low, the probabilities to sample other but the class with the highest log probability will be small, 
# and the model will probably output the most correct text, but rather boring, with small variation.

chat = ChatOpenAI(temperature=.1, openai_api_key=os.getenv("OPENAI_API_KEY"))
print(chat(
    [
        SystemMessage(content="You are a nice AI bot that helps a user figure out where to travel in one short sentence"),
        HumanMessage(content="I like the beaches where should I go?"),
        AIMessage(content="You should go to Nice, France"),
        HumanMessage(content="What else should I do when I'm there?")
    ]
).content)