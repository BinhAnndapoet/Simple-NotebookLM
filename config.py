from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    data_dir: Path = Path("data")
    storage_dir: Path = Path("storage/qdrant")
    qdrant_collection: str = "rag_chunks"

    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)
    top_k: int = Field(default=5, ge=1, le=64)

    embedding_model: str = "keepitreal/vietnamese-sbert"
    
    llm_provider: Literal["hf_local", "gemini", "vllm"] = "hf_local"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    hf_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    hf_device: int = 0
    hf_max_new_tokens: int = Field(default=1024, ge=1)

    gemini_model: str = "gemini-2.5-flash"
    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")

    summarize_batch_size: int = Field(default=10, ge=1)
    summarize_retrieval_k: int = Field(default=12, ge=1, le=128)
    generation_retrieval_k: int = Field(default=16, ge=1, le=128)
    quiz_default_count: int = Field(default=5, ge=1, le=50)
    flashcards_default_count: int = Field(default=5, ge=1, le=100)
    api_url: str = "http://localhost:8000"

settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.storage_dir.mkdir(parents=True, exist_ok=True)

@model_validator(mode="after")
def validate_config(self) -> Settings:
    if self.chunk_overlap >= self.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")
    
    if self.hf_device < -1:
        raise ValueError("hf_device must be -1 for CPU or >= 0 for CUDA.")
    
    if self.llm_provider == "gemini" and not self.google_api_key:
        raise ValueError("GOOGLE_API_KEY is required when llm_provider='gemini'.")  
    return self