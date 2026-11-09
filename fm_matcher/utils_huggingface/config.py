import os

default_config = {
    "QUERY_HUGGINGFACE": True,  # if set to False, this generates a random result instead of prompting the LLM
    "HUGGINGFACE_MODEL": "Qwen/Qwen3-1.7B",  # the model to use (local path or HF Hub ID)
    "HUGGINGFACE_N": 1,  # the number of answers to generate per prompt
    "HUGGINGFACE_TEMPERATURE": 0.7,  # the model temperature to use
    "HUGGINGFACE_TIMEOUT": 300,  # the timeout for inference (if we were using an API, but good to keep for consistency)
    "TEMPLATE_DIR": "resources/prompt_templates",  # the directory where prompt templates are stored
    "SQLITE_PATH": "huggingface_results_1-N.sqlite3",  # the path to the SQLite database file. Set this to None to disable storage.
    "HUGGINGFACE_BATCH_SIZE": 64,  # chunk size for concurrent batch inference (64-256 for optimal GPU utilization)
    "HUGGINGFACE_MAX_CONCURRENT": 4,  # max concurrent batch operations (tune down if OOM, up to 64 for better utilization)
}

config = {
    # use the environment variables if set (typecasted to the type given in default_config), other use default_config values
    k: None if os.getenv(k, v) == "None" else type(default_config[k])(os.getenv(k, v))
    for k, v in default_config.items()
}
