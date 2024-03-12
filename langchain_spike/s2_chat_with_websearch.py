from dotenv import find_dotenv, load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools

from helpers import get_gpt35_chat_client

load_dotenv(find_dotenv())

def main():
    llm = get_gpt35_chat_client()
    tools = load_tools(["serpapi"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True
    )
    agent.run("What is the difference in ages between main candidates in US election?")

if __name__ == "__main__":
    main()