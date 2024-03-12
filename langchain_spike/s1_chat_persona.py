from helpers import get_gpt35_chat_client
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.callbacks import get_openai_callback


def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.invoke(query)
        print(f'Spent a total of {cb.total_tokens} tokens')
    return result


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer. with the following conversation: {history}"),
    ("user", "{input}")
])
memory = ConversationBufferMemory()
def main():
    chat = get_gpt35_chat_client()
    chain = ConversationChain(llm=chat, prompt=prompt, memory=memory)
    response = count_tokens(chain, {"input": "hello"})
    print(response.content)

if __name__ == '__main__':
    main()

