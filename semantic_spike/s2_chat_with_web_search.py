'''
SPIKE: Chat with web search
1. Validate that we can use web search in a chat
2. Search for the answer if you can't answer the question directly
'''
import asyncio
import os

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai

from dotenv import load_dotenv
from semantic_kernel.connectors.search_engine.bing_connector import BingConnector
from semantic_kernel.core_plugins.web_search_engine_plugin import WebSearchEnginePlugin
from semantic_kernel.functions import KernelArguments
from semantic_kernel.prompt_template.kernel_prompt_template import KernelPromptTemplate
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig


load_dotenv()
GPT_35_CHAT_SERVICE_ID = "azure-chat-gpt-35"
DEPLOYMENT, API_KEY, ENDPOINT = sk.azure_openai_settings_from_dot_env()
BING_API_KEY = os.getenv("BING_API_KEY")


def kernel_with_chat():
    kernel = sk.Kernel()
    gpt_35_chat_service = sk_oai.AzureChatCompletion(
        service_id=GPT_35_CHAT_SERVICE_ID,
        api_key=API_KEY,
        endpoint=ENDPOINT,
        deployment_name=DEPLOYMENT,
    )
    kernel.add_service(gpt_35_chat_service)
    return kernel


# go straight to bing for everything
async def example1(kernel, search_plugin_name: str):
    print("==== BING PLUGIN ====")
    question = "What is the largest building in the world?"
    function = kernel.plugins[search_plugin_name]["search"]
    result = await kernel.invoke(function, query=question)

    print(f"Question: {question}")
    print(f"--- {search_plugin_name} ---")
    print(f"Answer: {result}")


# check first if you know the answer, then hit bing
async def example2(kernel: sk.Kernel, service_id: str):
    print("======== Use the Search Plugin to Answer User Questions ========")

    prompt = """
    Answer questions only when you know the facts or the information is provided.
    When you don't have sufficient information you reply with a list of commands to find the information needed.
    When answering multiple questions, use a bullet point list.
    Note: make sure single and double quotes are escaped using a backslash char.

    [COMMANDS AVAILABLE]
    - bing.search

    [INFORMATION PROVIDED]
    {{ $externalInformation }}

    [EXAMPLE 1]
    Question: what's the biggest lake in Italy?
    Answer: Lake Garda, also known as Lago di Garda.

    [EXAMPLE 2]
    Question: what's the biggest lake in Italy? What's the smallest positive number?
    Answer:
    * Lake Garda, also known as Lago di Garda.
    * The smallest positive number is 1.

    [EXAMPLE 3]
    Question: what's Ferrari stock price? Who is the current number one female tennis player in the world?
    Answer:
    {{ '{{' }} bing.search ""what\\'s Ferrari stock price?"" {{ '}}' }}.
    {{ '{{' }} bing.search ""Who is the current number one female tennis player in the world?"" {{ '}}' }}.

    [END OF EXAMPLES]

    [TASK]
    Question: {{ $question }}.
    Answer:
    """
    question = "Who is the most followed person on TikTok right now? What's the exchange rate EUR:USD?"
    print(question)

    oracle = kernel.create_function_from_prompt(
        function_name="oracle",
        plugin_name="OraclePlugin",
        prompt_template=prompt,
        template_format="semantic-kernel",
        description="Answer questions only when you know the facts or the information is provided.",
        execution_settings=sk_oai.OpenAIChatPromptExecutionSettings(
            service_id=service_id, max_tokens=150, temperature=0, top_p=1
        ),
    )
    answer = await kernel.invoke(
        oracle,
        question=question,
        externalInformation="",
    )

    result = str(answer)

    if "bing.search" in result:
        prompt_template = KernelPromptTemplate(PromptTemplateConfig(template=result))

        print("--- Fetching information from Bing... ---")
        information = await prompt_template.render(kernel, KernelArguments())

        print("Information found:\n")
        print(information)

        answer = await kernel.invoke(oracle, question=question, externalInformation=information)
        print("\n---- Oracle's Answer ----:\n")
        print(answer)
    else:
        print("AI had all of the information, there was no need to query Bing.")



async def main():
    # Create a kernel with a chat service
    kernel = kernel_with_chat()

    # Add the bing plugin
    bing_connector = BingConnector(api_key=BING_API_KEY)
    bing_plugin = WebSearchEnginePlugin(bing_connector)
    kernel.import_plugin_from_object(bing_plugin, plugin_name="bing")

    await example1(kernel, "bing")
    # await example2(kernel, GPT_35_CHAT_SERVICE_ID)


if __name__ == "__main__":
    asyncio.run(main())