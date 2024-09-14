import cohere
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

cohere_api_key = 'QTl4jayogr7OjDxTUn0b5eVGdC8fzBgcTlZlJzz1'
co = cohere.Client(cohere_api_key)

class CohereModel:
    def __init__(self, model, max_tokens):
        self.model = model
        self.max_tokens = max_tokens

    def __call__(self, prompt):
        response = co.generate(
            model=self.model,
            prompt=prompt,
            max_tokens=self.max_tokens
        )
        return response.generations[0].text.strip()

llm = CohereModel(model="command-r-plus", max_tokens=200)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a skilled environmental scientist who can provide detailed explanations on ecological and environmental issues."),
    ("human", "{question}")
])

class Sequence:
    def __init__(self, prompt, llm, output_parser):
        self.prompt = prompt
        self.llm = llm
        self.output_parser = output_parser

    def invoke(self, inputs):
        full_prompt = self.prompt.format(**inputs)
        llm_response = self.llm(full_prompt)
        return self.output_parser.parse(llm_response)

sequence = Sequence(prompt, llm, StrOutputParser())

response = sequence.invoke({"question": "Can you describe the water cycle and its importance to Earth's ecosystems"})
print("Response from LECL Chain")
print(response)