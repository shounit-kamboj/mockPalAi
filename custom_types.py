import pydantic

class RAGChunkAndSrc(pydantic.BaseModel):
    chunk: list[str]
    source_id: str = None

class RAGUpsertResult(pydantic.BaseModel):
   ingested: int

class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]

class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_context: int

class RagQuestion(pydantic.BaseModel):
    id: str
    question: str
    category: str
    topic: str
    skills: list[str]
    difficulty: str | None = None
    seniority: str | None = None
    source: str
    source_url: str
    license: str

class RagQuestions(pydantic.BaseModel):
    questions: list[RagQuestion]






