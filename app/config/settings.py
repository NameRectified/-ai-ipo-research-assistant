from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM Provider Priority ---
    llm_provider_priority: str = "groq,gemini,openrouter"

    # --- Groq ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # --- Gemini ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # --- OpenRouter ---
    openrouter_api_key: str = ""
    openrouter_models: str = (
        "google/gemini-2.0-flash-001,"
        "meta-llama/llama-3-70b-instruct,"
        "mistralai/mistral-7b-instruct"
    )

    # Paths
    model_path: str = "models/model.pkl"

    # Logging
    log_level: str = "INFO"


settings = Settings()