import os
import requests
from uuid import uuid4
from langsmith import Client
from langchain_community.llms import Cohere

unique_id = uuid4().hex[0:8]

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "XYZ")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY", "XYZ")

tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2")
endpoint = os.getenv("LANGCHAIN_ENDPOINT")
api_key = os.getenv("LANGCHAIN_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")

if not all([tracing_v2, endpoint, api_key, cohere_api_key]):
    raise ValueError("One or more environment variables are not set Please check LANGCHAIN_TRACING_V2, LANGCHAIN_ENDPOINT, LANGCHAIN_API_KEY, and COHERE_API_KEY")

dataset_inputs = [
    "Tell us about AWS Lambda service and its key features.",
    "How does Google Kubernetes Engine (GKE) simplify container management?",
    "Explain the benefits of using Azure DevOps for CI/CD.",
    "What are the main differences between Docker and Podman?"
]

dataset_outputs = [
    {"must_mention": ["serverless", "functions"]},
    {"must_mention": ["containers", "orchestration"]},
    {"must_mention": ["CI/CD", "pipelines"]},
    {"must_mention": ["containers", "compatibility"]}
]

client = Client()
dataset_name = "DifferentCloudServices"

try:
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Cloud Services QA.",
    )

    client.create_examples(
        inputs=[{"question": q} for q in dataset_inputs],
        outputs=dataset_outputs,
        dataset_id=dataset.id,
    )
    print("Dataset created successfully.")

except requests.exceptions.HTTPError as e:
    print(f"HTTP error occurred: {e}")
    print(f"Response text: {e.response.text}")

except Exception as e:
    print(f"An error occurred: {e}")

llm = Cohere(
    model="command",
    max_tokens=300,
    temperature=0.7,
)

prompt = "Explain the advantages of using Terraform for infrastructure as code"
response = llm(prompt)
print(response)
