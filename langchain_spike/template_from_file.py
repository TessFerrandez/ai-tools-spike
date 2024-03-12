from helpers import get_gpt35_chat_client
from langchain.prompts import load_prompt


def main():
    prompt = load_prompt("langchain_spike/templates/product_name_prompt.json")
    user_input = prompt.format(product="apples")
    chat = get_gpt35_chat_client()
    response = chat.invoke(user_input)
    print(response.content)


if __name__ == "__main__":
    main()
