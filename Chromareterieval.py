from langchain.chains import RetrievalQA
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_cohere.llms import Cohere
from tenacity import retry, stop_after_attempt, wait_exponential
from requests.exceptions import RequestException, HTTPError
import os
import time
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv('COHERE_API_KEY', 'XYZ')

if not COHERE_API_KEY:
    print("Error: COHERE_API_KEY environment variable not found.")
    exit(1)
else:
    print(f"COHERE_API_KEY loaded successfully: {COHERE_API_KEY[:5]}...")

llm = Cohere(temperature=0, cohere_api_key=COHERE_API_KEY)
embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)

client = chromadb.HttpClient(host="127.0.0.1", port=8000)

db = Chroma(persist_directory="./chromadb", embedding_function=embeddings, client=client)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

docs = retriever.get_relevant_documents("Tell us which module is most relevant to LLMs and Generative AI")

def pretty_print_docs(docs):
    print("\n" + "-" * 100 + "\n".join(f"Document {i + 1}:\n\n{d.page_content}" for i, d in enumerate(docs)) + "\n" + "-" * 100)
pretty_print_docs(docs)

for doc in docs:
    print(doc.metadata)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=20))
def get_answer_with_retries(query):
    try:
        return qa_chain({"query": query})
    except (RequestException, HTTPError) as e:
        print(f"Error making request to Cohere: {e}")
        raise

query = "Tell us which module is most relevant to LLMs and Generative AI"
response = get_answer_with_retries(query)
print(response)
