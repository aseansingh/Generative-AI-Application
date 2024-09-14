import cohere
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

cohere_api_key = 'XYZ'
co = cohere.Client(cohere_api_key)

class CohereModel:
    def __init__(self, model, max_tokens):
        self.model = model
        self.max_tokens = max_tokens

    def __call__(self, prompt):
        if isinstance(prompt, list):
            prompt = "\n".join(prompt)
        elif isinstance(prompt, dict):
            prompt = str(prompt)
        print(f"Generated prompt for Cohere: {prompt}")
        response = co.generate(
            model=self.model,
            prompt=prompt,
            max_tokens=self.max_tokens
        )
        return response.generations[0].text.strip()

llm = CohereModel(model="command-r-plus", max_tokens=200)

response = llm("Tell me something interesting about the ocean")
print("Case1 Response - > " + response)

template = """You are a chatbot having a conversation with a human.
Human: {human_input} + {city}
:"""

prompt = PromptTemplate(input_variables=["human_input", "city"], template=template)

prompt_val = prompt.invoke({"human_input": "Describe the best places to visit in", "city": "San Francisco"})
print("Prompt String is ->")
print(prompt_val)

class Chain:
    def __init__(self, prompt_template, llm):
        self.prompt_template = prompt_template
        self.llm = llm

    def invoke(self, inputs):
        prompt = self.prompt_template.invoke(inputs)
        if hasattr(prompt, 'to_string'):
            prompt = prompt.to_string()
        return self.llm(prompt)

chain = Chain(prompt, llm)

response = chain.invoke({"human_input": "Describe the best places to visit in", "city": "San Francisco"})

print("Case2 Response - >" + response)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an enthusiastic tour guide who loves sharing information in detail."),
        ("ai", "I am excited to share detailed information in steps!"),
        ("human", "{input}"),
    ]
)

chain = Chain(prompt, llm)
response = chain.invoke({"input": "Can you tell me an interesting fact about the Eiffel Tower?"})
print("Case3 Response - > " + response)

prompt = ChatPromptTemplate.from_template("Share a fun fact about {animal}")
chain1 = Chain(prompt, llm)
response = chain1.invoke({"animal": "dolphins"})
print("Case4 Response - > " + response)

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful assistant that re-writes the user's text to "
                "sound more confident."
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)

chain2 = Chain(chat_template, llm)
response = chain2.invoke({"text": "Are you sure if I can complete this project on time"})
print("Case5 Response ->")
print(response)
