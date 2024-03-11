import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai


CHAT_SERVICE_ID = "default"
DEPLOYMENT, API_KEY, ENDPOINT = sk.azure_openai_settings_from_dot_env()


def kernel_with_chat():
    kernel = sk.Kernel()
    chat_service = sk_oai.AzureChatCompletion(
        service_id=CHAT_SERVICE_ID,
        api_key=API_KEY,
        endpoint=ENDPOINT,
        deployment_name=DEPLOYMENT,
    )
    kernel.add_service(chat_service)
    return kernel
