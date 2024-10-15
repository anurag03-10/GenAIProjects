#This program is intended to demo the use of following:
# 1. Web Base Loader to read a webpage
# 2. RecursiveCharacterTextSplitter to chunk the content into documents.
# 3. Convert the document into embedding and store it into FAISS DB.
# 4. Create a stuff document chain, Create a retrival chain from FAISS DB.
# 5. Create a retrieval chain using FAISS retriever and document chain .

from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain

loader = WebBaseLoader("https://code4x.dev/courses/chat-app-using-langchain-openai-gpt-api-pinecone-vector-database/")

docs=loader.load()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

llm=GoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key="AIzaSyByc5nOHPy51tTMUhoN-c2xoB99VQSFIJc")

embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyByc5nOHPy51tTMUhoN-c2xoB99VQSFIJc")

vector= FAISS.from_documents(documents,embeddings)

prompt =  ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:
                                           
<Context>
    {context}
</Context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm,prompt)

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain) #document chain being part of retrieval chain

response =  retrieval_chain.invoke({
    "context": "YOu are the trainer who is teaching the given course and you have to suggest the potential learners",
    "input": "What are the key takeaways for learners from this Course?"
})

print(response["answer"])