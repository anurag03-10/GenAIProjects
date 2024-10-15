from langchain_community.chat_models import ChatOpenAI
import google.generativeai as genai
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import conversational_retrieval
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

loader = WebBaseLoader("https://medium.com/swlh/algorithmic-management-what-is-it-and-whats-next-33ad3429330b")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()

documents = text_splitter.split_documents(docs)

llm= ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
# llm= ChatOpenAI(model="ggpt-3.5-turbo")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector = FAISS.from_documents(documents,embeddings)

prompt = ChatPromptTemplate.from_template(
    """ Answer the following question based only on the provided context:

"""
)