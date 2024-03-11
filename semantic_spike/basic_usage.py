import asyncio
from helpers import kernel_with_chat


async def main():
    kernel = kernel_with_chat()
    fun_plugin = kernel.import_plugin_from_prompt_directory("semantic_spike/Plugins", "FunPlugin")
    joke_function = fun_plugin["Joke"]
    joke_result = await kernel.invoke(joke_function, input="elephants", style="silly")
    print(joke_result)


if __name__ == "__main__":
    asyncio.run(main())
