from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    
    # Model provider configuration
    api_key: str | None = None
    api_base: str = "https://api.deepseek.com/v1"
    
    # Model configuration
    model_name: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # MongoDB configuration
    mongodb_uri: str = "mongodb://mongodb:27017"
    mongodb_database: str = "manus"
    mongodb_username: str | None = None
    mongodb_password: str | None = None
    
    # Redis configuration
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    
    # Sandbox configuration
    sandbox_address: str | None = None
    sandbox_image: str | None = None
    sandbox_name_prefix: str | None = None
    sandbox_ttl_minutes: int | None = 30
    sandbox_network: str | None = None  # Docker network bridge name
    sandbox_chrome_args: str | None = ""
    sandbox_https_proxy: str | None = None
    sandbox_http_proxy: str | None = None
    sandbox_no_proxy: str | None = None
    
    # Search engine configuration
    search_provider: str | None = None  # "google", "baidu"
    google_search_api_key: str | None = None
    google_search_engine_id: str | None = None
    
    # MCP configuration
    mcp_config_path: str = "/etc/mcp.json"
    
    # Logging configuration
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        
    def validate(self):
        if not self.api_key:
            raise ValueError("API key is required")

@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    settings.validate()
    return settings

# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-pro"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    tools = [
        types.Tool(url_context=types.UrlContext()),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.6,
        thinking_config = types.ThinkingConfig(
            thinking_budget=31179,
        ),
        tools=tools,
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()
