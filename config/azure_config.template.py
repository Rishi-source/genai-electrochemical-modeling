AZURE_CONFIG = {
    "api_version": "2024-12-01-preview",
    "azure_endpoint": "YOUR_AZURE_ENDPOINT_HERE",  
    "api_key": "YOUR_API_KEY_HERE",
    "deployment_name": "YOUR_DEPLOYMENT_NAME",  
    "model": "gpt-4",
    "temperature": 0.3,
    "max_tokens": 2000,
}


EMBEDDING_CONFIG = {
    "azure_endpoint": "YOUR_EMBEDDING_ENDPOINT_HERE",
    "api_key": "YOUR_EMBEDDING_API_KEY_HERE",
    "deployment_name": "text-embedding-3-large",
    "model": "text-embedding-3-large",
    "chunk_size": 1000,
}


PROMPT_CONFIG = {
    "system_role": "You are an expert in electrochemical engineering and computational modeling.",
    "max_history": 10,
    "temperature": 0.3,
    "top_p": 0.95,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}


RETRY_CONFIG = {
    "max_retries": 3,
    "initial_delay": 1.0,
    "exponential_base": 2.0,
    "max_delay": 60.0,
}


TOKEN_BUDGET = {
    "max_input_tokens": 128000,
    "max_output_tokens": 4096,
    "reserved_tokens": 1000,
}


def get_azure_client_config():
    return {
        "api_version": AZURE_CONFIG["api_version"],
        "azure_endpoint": AZURE_CONFIG["azure_endpoint"],
        "api_key": AZURE_CONFIG["api_key"],
    }


def get_chat_completion_params():
    return {
        "deployment_name": AZURE_CONFIG["deployment_name"],
        "temperature": PROMPT_CONFIG["temperature"],
        "max_tokens": AZURE_CONFIG["max_tokens"],
        "top_p": PROMPT_CONFIG["top_p"],
        "frequency_penalty": PROMPT_CONFIG["frequency_penalty"],
        "presence_penalty": PROMPT_CONFIG["presence_penalty"],
    }


def get_embedding_params():
    return {
        "deployment_name": EMBEDDING_CONFIG["deployment_name"],
        "model": EMBEDDING_CONFIG["model"],
    }
