from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class LLMManager:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

    def invoke(self, prompt: ChatPromptTemplate, **kwargs) -> str:
        messages = prompt.format_messages(**kwargs)
        response = self.llm.invoke(messages)
        return str(response.content)