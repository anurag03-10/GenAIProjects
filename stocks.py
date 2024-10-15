import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import create_retrieval_chain

API_KEY= 'AIzaSyD1pPu85DAKaFKRJi4ASb4gTvUlwdSxQGs'
SEARCH_ENGINE_ID= 'd1bf03b5898ee4339'
GOOGLE_API_KEY='AIzaSyByc5nOHPy51tTMUhoN-c2xoB99VQSFIJc'
# Function to build the payload for Google Search API
def build_payload(query, start=1, num=10, date_restrict='d1', **params):
    payload = {
        'key': API_KEY,
        'q': query,
        'cx': SEARCH_ENGINE_ID,
        'start': start,
        'num': num,
        'dateRestrict': date_restrict
    }
    payload.update(params)
    return payload

# Function to make a request to Google Search API
def make_request(payload):
    response = requests.get('https://www.googleapis.com/customsearch/v1', params=payload)
    if response.status_code != 200:
        raise Exception('Request Failed')
    return response.json()

# Function to perform a search and get URLs from Google Custom Search API
def search_urls(query, result_total=10):
    items = []
    reminder = result_total % 10
    if reminder > 0:
        pages = (result_total // 10) + 1
    else:
        pages = (result_total // 10)

    for i in range(pages):
        if pages == i + 1 and reminder > 0:
            payload = build_payload(query, start=(i + 1) * 10, num=reminder)
        else:
            payload = build_payload(query, start=(i + 1) * 10)
        response = make_request(payload)
        items.extend(response['items'])
    
    # Extract the URLs from the search results
    urls = [item['link'] for item in items]
    print(urls[:5])
    return urls[:5]  # Return top 5 URLs

# Get user's search query
user_query = "Best cardiologist in pune"

# Perform search and get URLs
urls = search_urls(user_query)

# Load the webpage content from the retrieved URLs
documents = []
for url in urls:
    loader = WebBaseLoader(url)
    docs = loader.load()
    documents.extend(docs)
# print(documents)
# Split the content into smaller chunks
text_splitter = RecursiveCharacterTextSplitter()
split_documents = text_splitter.split_documents(documents)

# Initialize the language model and embeddings
llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Create FAISS vector store from the documents and embeddings
vector = FAISS.from_documents(split_documents, embeddings)

# Define the prompt template with correct input variable names
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:

<context>
    {context}
</context>

Question: {input}""")

# Create the document chain and retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Enhanced context with a powerful instructor role
context = f"""
You are an advanced AI instructor with deep expertise in various domains. 
Your task is to search the web and analyze the most relevant and credible sources from top URLs to provide a well-researched and accurate answer.
Your response should be concise, insightful, and directly address the user's query based on the latest information found in the articles.
Ensure the answer is authoritative and covers key aspects of the topic.

User Query: {user_query}
"""

# Invoke the retrieval chain using the enhanced context
response = retrieval_chain.invoke({
    "context": context,
    "input": f"Based on the latest information available online, what are the key insights on {user_query}?"
})

print(response["answer"])
