from dotenv import load_dotenv
import logging
import sys
import os

from langchain.document_loaders import PyPDFLoader # for loading the pdf
from langchain.llms import OpenAI # the LLM model we'll use (CHatGPT)
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
import json
from langchain.docstore.wikipedia import Wikipedia
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.agents.react.base import DocstoreExplorer
from langchain import PromptTemplate

def test_react_docstore_agent():
    # build tools
    docstore=DocstoreExplorer(Wikipedia())
    tools = [
        Tool(
            name="Search",
            func=docstore.search,
            description="Search for a term in the docstore.",
        ),
        Tool(
            name="Lookup",
            func=docstore.lookup,
            description="Lookup a term in the docstore.",
        )
    ]
    # build LLM
    llm = OpenAI(
        model_name="text-davinci-003",
        temperature=0,
    )
    # initialize ReAct agent
    react = initialize_agent(tools, llm, agent="react-docstore", verbose=True)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=react.agent,
        tools=tools,
        verbose=True,
        max_iterations=2
    )
    # perform question-answering
    question = "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?"
    agent_executor.run(question)


def test_agent():
    openai_api_key=os.getenv("OPENAI_API_KEY")
    google_api_key=os.getenv("GOOGLE_API_KEY")
    google_cse_id=os.getenv("GOOGLE_CSE_ID")
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    toolkit = load_tools(
        ["google-search"], llm=llm, 
        google_api_key=google_api_key, google_cse_id=google_cse_id)
    
    agent = initialize_agent(
        toolkit, llm, agent="zero-shot-react-description", verbose=True, return_intermediate_steps=True)
    response = agent({
        "input":"what was the first album of the band that Natalie Bergman is a part of?",
        "max_iterations": 1})
    print(response)


def test_prompt_template():
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=.0, openai_api_key=os.getenv("OPENAI_API_KEY"))

    template = """
    I really want to support the {team}. What should I do as a fan?

    Respond in one short sentence
    """

    prompt = PromptTemplate(
        input_variables=["team"],
        template=template,
    )

    final_prompt = prompt.format(team='Manchester United')

    print(f"Final Prompt: {final_prompt}")
    print("-----------")
    print(f"test_prompt_template Output: {llm(final_prompt)}")


def test_chat_messages():
    import langchain
    langchain.debug = True
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage, AIMessage
    result = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=.0, openai_api_key=os.getenv("OPENAI_API_KEY"))(
        [
            SystemMessage(content="You are a helpful AI assistant specialising in data engineering, providing concise advice to users"),
            HumanMessage(content="""
                BookMyEvent Inc is an online event planning and management company that wants to improve its analytics capabilities. 
                BookMyEvent Inc currently utilizes multiple microservices written in Python and Golang. These microservices, such as the UserManagement service, are packaged as Docker containers and run on a Kubernetes cluster. The backend services are served via an API Gateway, and the UserManagement microservice handles user lifecycle and authentication using Python. The Inventory microservice, written in Go, manages the purchasing process, and the Inventory DB stores both product information and purchase history. The company also collects clickstream events from the frontend using Snowplow in Kafka. In terms of external marketing sources, BookMyEvent utilizes Google, Facebook, Tiktok, and Twitter Ads.
                BookMyEvent Inc is seeking a data platform solution to enhance their analytics capabilities. The solution should be implemented on a cloud platform and enable reporting, A/B testing, ad-hoc research, and potentially machine learning models. It should also be efficient and scalable to handle large data sources. Consistency of data used for reporting is important, and the solution should provide historical changes for non-append tables. 
                The deliverables include a 30-minute presentation explaining the solution architecture, design choices, and how the soft requirements were addressed. 
                Assumptions should be documented, an architecture diagram using official cloud icons is recommended.
            """.replace("                ", '').replace("\n", '')),
            AIMessage(content="Cloud Platform: Utilize a cloud platform such as Amazon Web Services (AWS) or Google Cloud Platform (GCP) to build the data platform solution."),
            HumanMessage(content="Can you organise AWS, GCP and AZURE from highest to lowest cost and make an estimate for an example of use with the architecture provided?"),
        ]
    )
    print("-----------")
    print(f"test_chat_messages Output: {str(result.content)}")


def test_chat_messages_interview():
    import langchain
    langchain.debug = True
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage, AIMessage
    result = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=.0, openai_api_key=os.getenv("OPENAI_API_KEY"))(
        [
            SystemMessage(content="""
            You are a job interviewer AI specialising in data engineering. Your duty is to ask questions to the user based on the following use case:

            BookMyEvent Inc is an online event planning and management company that wants to improve its analytics capabilities. 
            BookMyEvent Inc currently utilizes multiple microservices written in Python and Golang. These microservices, such as the UserManagement service, are packaged as Docker containers and run on a Kubernetes cluster. The backend services are served via an API Gateway, and the UserManagement microservice handles user lifecycle and authentication using Python. The Inventory microservice, written in Go, manages the purchasing process, and the Inventory DB stores both product information and purchase history. The company also collects clickstream events from the frontend using Snowplow in Kafka. In terms of external marketing sources, BookMyEvent utilizes Google, Facebook, Tiktok, and Twitter Ads.
            BookMyEvent Inc is seeking a data platform solution to enhance their analytics capabilities. The solution should be implemented on a cloud platform and enable reporting, A/B testing, ad-hoc research, and potentially machine learning models. It should also be efficient and scalable to handle large data sources. Consistency of data used for reporting is important, and the solution should provide historical changes for non-append tables. 
            The deliverables include a 30-minute presentation explaining the solution architecture, design choices, and how the soft requirements were addressed. 
            Assumptions should be documented, an architecture diagram using official cloud icons is recommended.
            """),
            HumanMessage(content="""
            The solution should have:
            Cloud-native services: Leveraging cloud-managed services ensures scalability, reliability, and cost-effectiveness.
            Stream processing: Real-time data processing with Kafka allows for immediate insights into user activity and marketing campaign performance.
            Data lake and data warehouse: Separating raw data storage from structured data warehousing enables flexibility and scalability.
            Time-series database: Storing clickstream data in a time-series database allows for efficient historical data retrieval.
            Data governance: Implementing data lineage and cataloging tools to ensure data consistency and traceability.
            """)
        ]
    )
    print("-----------")
    print(f"test_chat_messages_interview Output:\n {str(result.content)}")


def test_chat_messages_interview_answer():
    import langchain
    langchain.debug = True
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage, AIMessage
    result = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=.0, openai_api_key=os.getenv("OPENAI_API_KEY"))(
        [
            SystemMessage(content="""
            You are a helpful AI assistant specialised in data engineering, who fulfils the role of doing a job interview. Your duty is to answer the user's questions avoiding use questions as answers based on the following use case::
            BookMyEvent Inc is an online event planning and management company that wants to improve its analytics capabilities. 
            BookMyEvent Inc currently utilizes multiple microservices written in Python and Golang. These microservices, such as the UserManagement service, are packaged as Docker containers and run on a Kubernetes cluster. The backend services are served via an API Gateway, and the UserManagement microservice handles user lifecycle and authentication using Python. The Inventory microservice, written in Go, manages the purchasing process, and the Inventory DB stores both product information and purchase history. The company also collects clickstream events from the frontend using Snowplow in Kafka. In terms of external marketing sources, BookMyEvent utilizes Google, Facebook, Tiktok, and Twitter Ads.
            BookMyEvent Inc is seeking a data platform solution to enhance their analytics capabilities. The solution should be implemented on a cloud platform and enable reporting, A/B testing, ad-hoc research, and potentially machine learning models. It should also be efficient and scalable to handle large data sources. Consistency of data used for reporting is important, and the solution should provide historical changes for non-append tables. 
            The deliverables include a 30-minute presentation explaining the solution architecture, design choices, and how the soft requirements were addressed. 
            Assumptions should be documented, an architecture diagram using official cloud icons is recommended.
            """),
            HumanMessage(content="""
            2. Are there any specific cloud-native services or technologies that you would like to leverage for scalability and efficiency?
            3. Can you provide more details about the data sources and volumes that need to be handled by the data platform solution?
            4. How frequently do you expect data updates or changes to occur in the system?
            5. Are there any specific reporting or analytics tools that you would like to integrate with the data platform solution?
            6. Can you provide more information about the desired A/B testing capabilities? What kind of experiments do you plan to run?
            7. Are there any specific machine learning models that you would like to incorporate into the data platform solution?
            8. Can you provide more details about the desired ad-hoc research capabilities? What kind of research queries or analysis do you expect to perform?
            9. Are there any specific security or compliance requirements that need to be considered in the design of the data platform solution?
            10. Do you have any preferences or constraints regarding the programming languages or frameworks to be used in the implementation of the data platform solution?
            """),
            # AIMessage(content="For a beach-themed wedding, consider seashell centerpieces and aqua-blue accents"),
            # HumanMessage(content="What other ideas do you have for a beach wedding?")
        ]
    )
    print("-----------")
    print(f"test_chat_messages Output: {str(result.content)}")

def read_pdf():
    pdf_path = f"{os.getcwd()}/docs/case_study.pdf"
    pdf_loader = PyPDFLoader(pdf_path)
    content = pdf_loader.load()
    print(content)



def main():
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')

    # test_prompt_template()
    # test_chat_messages()
    test_chat_messages_interview()

if __name__ == '__main__':
    main()
