import logging
import os
import uuid

import inngest
import inngest.fast_api
from dotenv import load_dotenv
from fastapi import FastAPI
from inngest.experimental import ai

# Removed unused imports: aiohttp.payload, openai...message, requests.adapters, datetime.
# They were never used, and aiohttp/requests aren't in pyproject.toml.

from data_loader import load_and_chunk_pdf, embed_text, read_questions_json
from vector_db import QdrantStorage
from custom_types import (
    RAGUpsertResult,
    RAGSearchResult,
    RAGChunkAndSrc,
    RagQuestion,
    RagQuestions,
)

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_mock_interviewer",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),  # lets steps return Pydantic models (output_type=...)
)

# How many questions to embed + upsert per Inngest step.
# Small enough to stay under OpenAI's per-request input limit, big enough to keep the step count low
# (4,055 questions / 200 = ~21 steps).
BATCH_SIZE = 200
QUESTIONS_COLLECTION = "questions"


# =====================================================================================
# NEW: ingest the question bank (run once, or again after you update questions.json)
# Trigger it from the Inngest dashboard (localhost:8288) with:
#   event name: rag/questions_ingest
#   data:       {"file_path": "data/mockpal-question-bank/questions.json"}
# =====================================================================================
@inngest_client.create_function(
    fn_id="RAG: ingest questions",
    trigger=inngest.TriggerEvent(event="rag/questions_ingest"),
)
async def rag_ingest_questions(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RagQuestions:
        json_path = ctx.event.data["file_path"]
        qs = read_questions_json(json_path)
        return RagQuestions(questions=qs)

    def _upsert(batch: list[RagQuestion]) -> RAGUpsertResult:
        # TODO: vectors  = embed_text([...the .question of each item in batch...])
        # TODO: ids      = [...the .id of each item...]
        #       (already UUIDv5 strings, so don't generate new ones like rag_ingest_pdf does;
        #        reusing them is what makes re-running this function update instead of duplicate)
        # TODO: payloads = [...each item's .model_dump()...]  (Qdrant wants plain dicts)

        vectors = embed_text([q.question for q in batch])
        ids = [q.id for q in batch]
        payloads = [q.model_dump() for q in batch]
        QdrantStorage(collection=QUESTIONS_COLLECTION).upsert(ids, vectors, payloads)
        return RAGUpsertResult(ingested=len(batch))


    # Step 1: load + validate the file (one checkpoint)
    bank = await ctx.step.run("load_questions", lambda: _load(ctx), output_type=RagQuestions)

    # Step 2..N: one step per batch, so a failure only retries that batch
    total = 0
    for i in range(0, len(bank.questions), BATCH_SIZE):
        batch = bank.questions[i : i + BATCH_SIZE]
        result = await ctx.step.run(
                  f"embed_batch_{i}",            # every step needs a UNIQUE name
                  lambda b=batch: _upsert(b),    # b=batch binds THIS batch (late-binding trap)
                  output_type=RAGUpsertResult,
              )
        total += result.ingested


    return {"ingested": total}


# =====================================================================================
# OLD: PDF ingest. Keep until the new flow works, then delete
# (resumes are not stored in Qdrant in the new design)
# =====================================================================================
@inngest_client.create_function(
    fn_id="RAG: ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/pdf_ingest"),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunk=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunk
        source_id = chunks_and_src.source_id
        vecs = embed_text(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payload = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payload)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load_and_chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed_and_upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)

    return ingested.model_dump()


# =====================================================================================
# OLD: PDF query. This becomes the interview function in steps 3-5:
#   event data: resume + JD (instead of "question")
#   search:     the "questions" collection (instead of "docs")
#   prompt:     retrieved questions + full resume/JD, ask for tailored questions with source ids
# Keep as-is until the new interview function works.
# =====================================================================================
@inngest_client.create_function(
    fn_id="RAG: Query_ PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int):
        query_vector = embed_text([question])[0]
        store = QdrantStorage()
        found = store.search(query_vector, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = ctx.event.data.get("top_k", 30)  # the default lives HERE, not in _search

    found = await ctx.step.run("search", lambda: _search(question, top_k), output_type=RAGSearchResult)
    context_block = "\n\n".join(f"-{c}" for c in found.contexts)
    user_content = (
        "use the following context to interview the user like you are an expert hiring manager in this profession.\n\n"
        f"{context_block}\n\n"
        f"user question: {question}\n"  # fixed: newline so the next sentence doesn't run into the question
        "ask questions in a professional and serious manner"
    )

    adapter = ai.openai.Adapter(auth_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
    response = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You are an interviewer who only asks questions based on the context provided."},
                {"role": "user", "content": user_content},
            ],
        },
    )

    answer = response["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}


app = FastAPI()

# Every function must be listed here or Inngest won't see it.
# After editing: restart uvicorn, then check the dashboard lists "RAG: ingest questions".
inngest.fast_api.serve(app, inngest_client, [rag_ingest_questions, rag_ingest_pdf, rag_query_pdf_ai])