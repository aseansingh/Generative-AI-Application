import os
import time
from functools import lru_cache
from requests.exceptions import HTTPError, RequestException

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_cohere import CohereEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential
import cohere

from dotenv import load_dotenv
load_dotenv()

PDF_DOCS_DIRECTORY = "./pdf-docs"
PERSIST_DIRECTORY = "./chromadb"
COHERE_API_KEY = "XYZ"

if not COHERE_API_KEY:
    print("Error: COHERE_API_KEY environment variable not found or empty.")
    exit(1)
else:
    print(f"COHERE_API_KEY loaded successfully: {COHERE_API_KEY[:5]}...")

REQUESTS_PER_MINUTE = 5
REQUESTS_PER_HOUR = 250
BATCH_SIZE = 5
RETRY_ATTEMPTS = 10
RETRY_MIN_WAIT = 4
RETRY_MAX_WAIT = 120
SERVER_SIDE_RETRY_WAIT = 300

loader = PyPDFDirectoryLoader(PDF_DOCS_DIRECTORY)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)
print(f"Total number of documents: {len(split_docs)}")

embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)

db = Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIRECTORY)


def rate_limit_decorator(func):
    def wrapper(*args, **kwargs):
        time.sleep(1 / REQUESTS_PER_MINUTE)
        return func(*args, **kwargs)

    return wrapper

@lru_cache(maxsize=1000)
def embed_document(document_content):
    return embeddings.embed_query(document_content)

document_batches = [split_docs[i : i + BATCH_SIZE] for i in range(0, len(split_docs), BATCH_SIZE)]
for i, document_batch in enumerate(document_batches):
    print(f"Processing batch {i + 1}/{len(document_batches)}")
    documents_to_add = []
    for j, document in enumerate(document_batch):
        print(f"Processing document {i * BATCH_SIZE + j + 1}/{len(split_docs)}")

        @retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT))
        @rate_limit_decorator
        def process_with_retry():
            try:
                embedding = embed_document(document.page_content)
                document.metadata['embedding'] = embedding
                documents_to_add.append(document)
            except (RequestException, HTTPError, cohere.error.CohereError) as e:
                if isinstance(e, HTTPError) and e.response.status_code == 429:
                    print(f"Rate limit hit. Waiting for {SERVER_SIDE_RETRY_WAIT} seconds...")
                    time.sleep(SERVER_SIDE_RETRY_WAIT)
                else:
                    print(f"Unexpected error: {e}. Retrying...")
                raise

        process_with_retry()

    add_documents_with_rate_limit(documents_to_add)
db.persist()
