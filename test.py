import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# print(os.getenv("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(model='gemini-pro', google_api_key=os.getenv("GOOGLE_API_KEY"))
response=llm.invoke("Who is mahatma gandhi")
print(response.content)