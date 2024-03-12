"""
SPIKE: Chat persona
1. Create a simple chat persona with a system message
2. Add history - truncate or summarize history -- NOTE: This is not implemented yet in semantic kernel
3. Seamlessly switch between chat models
"""

import asyncio
import os
from typing import Any, List, Optional
import aiohttp
from langchain_core.callbacks import CallbackManagerForLLMRun

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain.chains import LLMChain
from langchain_core.prompts import (
    PromptTemplate,
)
from langchain_core.messages import BaseMessage, AIMessage
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryMemory,
)


class ApigeeChatBisonModel(BaseChatModel):
    service_id: str = "apigee-chat-bison"
    ai_model_id: str = "chat-bison"
    endpoint: str = os.environ.get("APIGEE_BASE_URL")
    headers: dict = {
        "Content-Type": "application/json",
        "x-apikey": os.environ.get("APIGEE_API_KEY"),
    }

    async def do_post(self, body: dict) -> AIMessage:
        print(body)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.endpoint}/chat-bison:predict",
                headers=self.headers,
                json=body,
                ssl=False,
            ) as response:
                response_json = await response.json()
        print(response_json)
        content = response_json["predictions"][0]["candidates"][0]["content"]
        return AIMessage(content=content)

    @property
    def _llm_type(self) -> str:
        return "ApigeeChatBisonModel"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        body = {
            "instances": [
                {
                    "messages": [
                        {"author": message.type, "content": message.content}
                        for message in messages
                    ]
                }
            ]
        }
        message = asyncio.run(self.do_post(body=body))

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])


template = """You are a nice chatbot having a conversation with a human.

Previous conversation:
{chat_history}

New human question: {input}
Response:"""
prompt = PromptTemplate.from_template(template)


def get_chat_with_buffer_memory(chat: BaseChatModel) -> LLMChain:
    memory = ConversationBufferMemory(memory_key="chat_history")
    return LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)


def get_chat_with_sliding_window_memory(chat: BaseChatModel) -> LLMChain:
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=1)
    return LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)


def get_chat_with_conversation_token_buffer_memory(chat: BaseChatModel) -> LLMChain:
    memory = ConversationTokenBufferMemory(
        memory_key="chat_history", llm=chat, max_token_limit=50
    )
    return LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)


def get_chat_with_summary_memory(chat: BaseChatModel) -> LLMChain:
    memory = ConversationSummaryMemory(memory_key="chat_history", llm=chat)
    return LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)


def main():
    model = ApigeeChatBisonModel()
    chain = get_chat_with_buffer_memory(model)
    for user_message in [
        "Hello, my name is Tom",
        "it's a lovely day in Royston, the rain is coming down.",
        "What's my name?",
    ]:
        chain.invoke({"input": user_message})


if __name__ == "__main__":
    main()
