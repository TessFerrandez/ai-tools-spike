# Can we have multiple chat services and switch between them?
# retry logic - ex. if you don't like the response, we can try a different model
# transparently switch between models as the conversation progresses - and the context window increases

# Add a system prompt to the chatbot

# Add history truncation (ex. drop oldest) / summarization
# Conditionally summarizing the previous context - ex. if the previous context is too long, summarize it
import asyncio

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai

from semantic_kernel.contents.chat_history import ChatHistory


GPT_35_CHAT_SERVICE_ID = "azure-chat-gpt-35"
DEPLOYMENT, API_KEY, ENDPOINT = sk.azure_openai_settings_from_dot_env()


def kernel_with_chat():
    kernel = sk.Kernel()
    chat_service = sk_oai.AzureChatCompletion(
        service_id=GPT_35_CHAT_SERVICE_ID,
        api_key=API_KEY,
        endpoint=ENDPOINT,
        deployment_name=DEPLOYMENT,
    )
    kernel.add_service(chat_service)
    return kernel

async def main():
    async def do_chat(user_input):
        print("User:", user_input)
        chat_result = await kernel.invoke(chat_function, user_input=user_input, history=chat_history)
        print("Bot:", chat_result)
        chat_history.add_user_message(user_input)
        chat_history.add_assistant_message(str(chat_result))


    # Create a kernel with a chat service
    kernel = kernel_with_chat()

    # Create a ChatHistory
    chat_history = ChatHistory()
    
    # Start chatting
    chat_plugin = kernel.import_plugin_from_prompt_directory("semantic_spike/Plugins", "ChatPlugin")
    chat_function = chat_plugin["Chat"]
    user_inputs = ["My name is Tess, what is your name?", "I love to read", "What is my name?"]
    for user_input in user_inputs:
        await do_chat(user_input)


if __name__ == "__main__":
    asyncio.run(main())