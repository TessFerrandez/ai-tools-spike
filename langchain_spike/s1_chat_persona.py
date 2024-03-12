from typing import List
from helpers import get_gpt35_chat_client
from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.memory import (
    ChatMessageHistory,
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryMemory,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import AzureOpenAI


from langchain_community.callbacks import get_openai_callback


def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.invoke(query)
        print(f"Spent a total of {cb.total_tokens} tokens")
    return result


template = """You are a nice chatbot having a conversation with a human.

Previous conversation:
{chat_history}

New human question: {input}
Response:"""
prompt = PromptTemplate.from_template(template)


def get_chat_with_buffer_memory():
    memory = ConversationBufferMemory(memory_key="chat_history")
    chat = get_gpt35_chat_client()
    return LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)


def get_chat_with_sliding_window_memory():
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=1)
    return chain_with_memory(memory)


def chain_with_memory(memory):
    chat = get_gpt35_chat_client()
    return LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)


def get_chat_with_conversation_token_buffer_memory():
    memory = ConversationTokenBufferMemory(
        memory_key="chat_history", llm=get_gpt35_chat_client(), max_token_limit=50
    )
    return chain_with_memory(memory)


def get_chat_with_summary_memory():
    model = get_gpt35_chat_client()
    memory = ConversationSummaryMemory(memory_key="chat_history", llm=model)
    return chain_with_memory(memory)


def main():
    model = get_gpt35_chat_client()
    memory = ConversationSummaryMemory(memory_key="chat_history", llm=model)
    chain = chain_with_memory(memory)

    for user_message in [
        "Hello, my name is Tom",
        "it's a lovely day in Royston, the rain is coming down.",
        "What's my name?",
    ]:
        response = count_tokens(chain, {"input": user_message})
        print(response)

        print("memory", memory.load_memory_variables({}))


if __name__ == "__main__":
    main()
