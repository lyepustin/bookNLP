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