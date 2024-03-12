from helpers import get_gpt35_chat_client
from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage


@tool
def get_word_length(word: str) -> int:
    ''' returns the length of a word '''
    return len(word)


def main():
    tools = [get_word_length]
    MEMORY_KEY = "chat_history"
    chat_history = []
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are very powerful assistant, but don't know current events",
            ),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    chat = get_gpt35_chat_client()
    chat_with_tools = chat.bind_tools(tools)
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]                
            ),
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt
        | chat_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    input1 = "how many letters are in the word educa?"
    result = agent_executor.invoke({
        "input": input1,
        "chat_history": chat_history
    })
    print("user:", input1)
    print("agent", result["output"])

    chat_history.extend([
        HumanMessage(input1),
        AIMessage(result["output"])
    ])
    input2 = "is that a real word?"
    result = agent_executor.invoke({
        "input": input2,
        "chat_history": chat_history
    })
    print("user:", input2)
    print("agent", result["output"])


if __name__ == '__main__':
    main()