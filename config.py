from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
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

    summarize_batch_size: int = Field(default=10, ge=1)
    summarize_retrieval_k: int = Field(default=12, ge=1, le=128)
    generation_retrieval_k: int = Field(default=16, ge=1, le=128)
    quiz_default_count: int = Field(default=5, ge=1, le=50)
    flashcards_default_count: int = Field(default=5, ge=1, le=100)

settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.storage_dir.mkdir(parents=True, exist_ok=True)