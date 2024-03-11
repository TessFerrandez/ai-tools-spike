'''
SPIKE: Chat persona
1. Create a simple chat persona with a system message
2. Add history - truncate or summarize history -- NOTE: This is not implemented yet in semantic kernel
3. Seamlessly switch between chat models
'''
import asyncio

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai

from semantic_kernel.contents.chat_history import ChatHistory


GPT_35_CHAT_SERVICE_ID = "azure-chat-gpt-35"
GPT_35_16K_CHAT_SERVICE_ID = "azure-chat-gpt-35-16k"
DEPLOYMENT, API_KEY, ENDPOINT = sk.azure_openai_settings_from_dot_env()


def kernel_with_chat():
    kernel = sk.Kernel()
    gpt_35_chat_service = sk_oai.AzureChatCompletion(
        service_id=GPT_35_CHAT_SERVICE_ID,
        api_key=API_KEY,
        endpoint=ENDPOINT,
        deployment_name=DEPLOYMENT,
    )
    gpt_35_16k_chat_service = sk_oai.AzureChatCompletion(
        service_id=GPT_35_16K_CHAT_SERVICE_ID,
        api_key=API_KEY,
        endpoint=ENDPOINT,
        deployment_name=DEPLOYMENT,
    )
    kernel.add_service(gpt_35_chat_service)
    kernel.add_service(gpt_35_16k_chat_service)
    return kernel


async def main():
    use_35=True

    async def do_chat(user_input):
        print("User:", user_input)
        if use_35:
            chat_result = await kernel.invoke(chat_35_function, user_input=user_input, history=chat_history)
        else:
            chat_result = await kernel.invoke(chat_35_16k_function, user_input=user_input, history=chat_history)
        print("Bot:", chat_result)
        chat_history.add_user_message(user_input)
        chat_history.add_assistant_message(str(chat_result))


    # Create a kernel with a chat service
    kernel = kernel_with_chat()

    # Create a ChatHistory
    chat_history = ChatHistory()
    
    # Start chatting
    chat_plugin = kernel.import_plugin_from_prompt_directory("semantic_spike/Plugins", "ChatPlugin")
    chat_35_function = chat_plugin["Chat35"]
    chat_35_16k_function = chat_plugin["Chat35_16k"]
    user_inputs = ["My name is Tess, what is your name?", "I love to read", "What is my name?"]
    for user_input in user_inputs:
        await do_chat(user_input)
        use_35 = not use_35


if __name__ == "__main__":
    asyncio.run(main())