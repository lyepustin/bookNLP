import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_SMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "default"

# Do not save inputs & outputs
os.environ["LANGCHAIN_HIDE_INPUTS"] = "true"
os.environ["LANGCHAIN_HIDE_OUTPUTS"] = "true"

# The below examples use the OpenAI API, so you will need
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
response = llm.invoke("Hello, world!")
print(response)