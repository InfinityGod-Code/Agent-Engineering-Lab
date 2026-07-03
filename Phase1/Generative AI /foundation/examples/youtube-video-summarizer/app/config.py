from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    # Add any configuration variables you need here
    GROQ_API_KEY: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


app_config = AppConfig()