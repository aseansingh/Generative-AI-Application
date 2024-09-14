from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import chromadb
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain_cohere.llms import Cohere
from langchain_cohere.embeddings import CohereEmbeddings

import os
from uuid import uuid4

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Test111 - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "XYZ"

COHERE_API_KEY = os.getenv('COHERE_API_KEY', 'XYZ')

if not COHERE_API_KEY:
    print("Error: COHERE_API_KEY environment variable not found.")
    exit(1)
else:
    print(f"COHERE_API_KEY loaded successfully: {COHERE_API_KEY[:5]}...")

llm = Cohere(
    temperature=0,
    cohere_api_key=COHERE_API_KEY
)

client = chromadb.HttpClient(host="127.0.0.1", settings=Settings(allow_reset=True))

embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)

db = Chroma(client=client, embedding_function=embeddings)

retv = db.as_retriever(search_type="similarity", search_kwargs={"k": 8})

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, output_key='answer')

qa = ConversationalRetrievalChain.from_llm(llm, retriever=retv, memory=memory, return_source_documents=True)

response = qa.invoke({"question": "What are the key features of the AWS Lambda service"})
print(memory.chat_memory.messages)

response = qa.invoke({"question": "How does Google Kubernetes Engine (GKE) simplify container management"})
print(memory.chat_memory.messages)

print(response)
