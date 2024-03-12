from helpers import get_gpt35_chat_client
from langchain.prompts import ChatPromptTemplate


def main():
    customer_email = """
    Arrr, I be fuming that me blender lid
    flew off and splattered me kitchen walls
    with smoothie! And to make matters worse,
    the warranty don't cover the cost of
    cleaning up me kitchen. I need yer help
    right now, matey!
    """

    customer_style = """
    Angry Danish, completely void of respect
    """

    customer_style = """
    American English
    in a calm and respectful tone
    """

    template_string = """Translate the text 
    that is delimited by triple backticks
    into a style that is {style}.
    text: ```{text}```
    """

    prompt_template = ChatPromptTemplate.from_template(template_string)
    user_input = prompt_template.format_messages(
        text=customer_email,
        style=customer_style
    )
    chat = get_gpt35_chat_client()
    response = chat.invoke(user_input)
    print(response.content)


if __name__ == "__main__":
    main()