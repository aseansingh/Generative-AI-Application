import cohere
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.llms.base import LLM
from langchain_core.outputs import LLMResult, Generation
from typing import Any, List, Mapping, Optional
import streamlit as st
from pydantic import Field

cohere_api_key = 'QTl4jayogr7OjDxTUn0b5eVGdC8fzBgcTlZlJzz1'
co = cohere.Client(cohere_api_key)


class CohereModel(LLM):
    model: str = Field(...)
    max_tokens: int = Field(...)

    class Config:
        arbitrary_types_allowed = True

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = co.generate(
            model=self.model,
            prompt=prompt,
            max_tokens=self.max_tokens,
            stop_sequences=stop or []
        )
        return response.generations[0].text.strip()

    @property
    def _llm_type(self) -> str:
        return "cohere"

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model, "max_tokens": self.max_tokens}


llm = CohereModel(model="command-r-plus", max_tokens=200)

history = StreamlitChatMessageHistory(key="chat_messages")

memory = ConversationBufferMemory(chat_memory=history, return_messages=True)

template = """You are an AI chatbot having a conversation with a human.
Human: {human_input}
AI: """
prompt_template = PromptTemplate(input_variables=["human_input"], template=template)

runnable_sequence = (
        {"human_input": RunnablePassthrough()}
        | prompt_template
        | llm
)

st.title('🦜🔗 Welcome to the ChatBot')

for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input("Enter your message here:"):
    st.chat_message("human").write(prompt)

    response = runnable_sequence.invoke(prompt)
    st.chat_message("ai").write(response)

    history.add_user_message(prompt)
    history.add_ai_message(response)