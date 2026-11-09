import os

default_config = {
    "QUERY_GEMINI": True,  # if set to False, this generates a random result instead of prompting the LLM
    "GEMINI_MODEL": "gemini-2.5-flash",  # the model to use on the Gemini API
    "GEMINI_N": 3,  # the number of answers to generate per prompt
    "GEMINI_TEMPERATURE": 1.0,  # the model temperature to use
    "GEMINI_TIMEOUT": 120,  # the timeout for Gemini API calls
    "TEMPLATE_DIR": "resources/prompt_templates",  # the directory where prompt templates are stored
    "GEMINI_REQUESTS_PER_MINUTE": 900,  # paid Gemini request limit (RMP)
    "GEMINI_TOKENS_PER_MINUTE": 900000,  # paid Gemini token limit (TMP)
    "GEMINI_RATE_LIMIT_BUFFER": 0.9,  # keep headroom below the configured rate cap
    "PARALLEL_GEMINI_REQUESTS": 1,  # the maximum number of parallel requests that will be sent to the Gemini API (lower this to fix frequent RateLimitErrors)
    "SQLITE_PATH": "dev.sqlite3",  # the path to the SQLite database file. Set this to None to disable storage.
}

config = {
    # use the environment variables if set (typecasted to the type given in default_config), other use default_config values
    k: None if os.getenv(k, v) == "None" else type(default_config[k])(os.getenv(k, v))
    for k, v in default_config.items()
}
