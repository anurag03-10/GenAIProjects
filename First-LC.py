from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    print(demosimple1.__doc__)
    demosimple()

def demosimple():
    prompt = ChatPromptTemplate.from_template("Tell me a few key achievement of {name}")
    model = GoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.5, google_api_key=os.getenv("GOOGLE_API_KEY"))
    chain = prompt | model
    print(chain.invoke({"name":"Abrahm Licoln"}))

def demosimple1():
    template = """
    Question : {question}
    Answer : 
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=['question']
    )

    question = "Which is most popular game in India"
    # llm= ChatOpenAI()
    llm= GoogleGenerativeAI(model="gemini-1.5-pro-latest")

    llm_chain = LLMChain(prompt=prompt,llm=llm)

    print(llm_chain.invoke(question)["text"])


def demosimple2():
    prompt = ChatPromptTemplate.from_template("Tell me a few key achievement of {name}")

    # model = ChatOpenAI()
    model = GoogleGenerativeAI(model="gemini-1.5-pro-latest")
    chain = prompt | model

    print(chain.invoke({"name":"Mahatma Gandhi"}))

demosimple()
