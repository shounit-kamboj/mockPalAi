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

class ResumeChunks(pydantic.BaseModel):
    chunks: list[str]      # small pieces, one search each (step 4)
    full_text: str         # whole resume, goes in the LLM prompt (step 5)


class RetrievedQuestion(RagQuestion):   # inherits all 10 question fields
    score: float                        # cosine similarity from Qdrant, higher = closer match


class RetrievedQuestions(pydantic.BaseModel):
    questions: list[RetrievedQuestion]

class InterviewQuestion(pydantic.BaseModel):
    source_id: str      # id of the bank question it was based on
    question: str       # the tailored question


class Interview(pydantic.BaseModel):
    questions: list[InterviewQuestion]






