
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest",google_api_key='AIzaSyByc5nOHPy51tTMUhoN-c2xoB99VQSFIJc')

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical document writer."),
    ("user","{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

output = chain.invoke ({"input":"How can langsmith help us with testing?"})

print(output)