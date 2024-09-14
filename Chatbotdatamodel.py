import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Cohere

os.environ["COHERE_API_KEY"] = "XYZ"

llm = Cohere(
    model="command",
    max_tokens=400,
    temperature=0.7,
)

embeddings = HuggingFaceEmbeddings()

vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

sample_questions = [
    "What is cloud computing?",
    "How does AWS differ from Azure?",
    "What are the benefits of using Google Cloud Platform?",
]

try:
    print("Running the chain on sample questions:")
    for question in sample_questions:
        result = chain({"query": question})
        print(f"\nQuestion: {question}")
        print(f"Answer: {result['result']}")

    print("\nChain run successfully.")

except Exception as e:
    print(f"An error occurred: {e}")
