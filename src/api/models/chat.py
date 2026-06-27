from pydantic import BaseModel, Field
from typing import List
from uuid import uuid4

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    chat_history: List[ChatMessage] = []

class Citation(BaseModel):
    article: str
    doc_id: str
    doc_title: str
    text_preview: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
