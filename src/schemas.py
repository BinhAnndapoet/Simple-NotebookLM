from pydantic import BaseModel, Field

class ChunkMetadata(BaseModel):
    document_id: str
    filename: str
    source: str
    page: int
    chunk_id: int
    section: str | None = None
    
class RetrievedChunk(BaseModel):
    text: str
    score: float
    metadata: ChunkMetadata
    
class Citation(BaseModel):
    source_index: int
    source_marker: str
    filename: str
    page: int
    section: str | None = None
    chunk_id: int | None = None

class RagAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)