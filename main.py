import logging

from aiohttp import payload
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime

from openai.types.beta.threads import message
from requests import adapters

from data_loader import load_and_chunk_pdf, embed_text
from vector_db import QdrantStorage
from custon_types import RAGUpsertResult, RAGSearchResult, RAGChunkAndSrc, RAGQueryResult

load_dotenv()

inngest_client = inngest.Inngest(
    app_id = "rag_mock_interviewer",
    logger = logging.getLogger("uvicorn"),
    is_production = False,
    serializer = inngest.PydanticSerializer() #lets you define types
)

@inngest_client.create_function(
    fn_id = "RAG: ingest PDF",
    trigger = inngest.TriggerEvent(event = "rag/pdf_ingest")
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


@inngest_client.create_function(
    fn_id = "RAG: Query_ PDF",
    trigger = inngest.TriggerEvent(event = "rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5):
        query_vector = embed_text([question])[0]
        store = QdrantStorage()
        found = store.search(query_vector, top_k)
        return RAGSearchResult(contexts = found["contexts"], sources = found["sources"] )

    question = ctx.event.data["question"]
    top_k = ctx.event.data.get("top_k", 5)

    found = await ctx.step.run("search", lambda: _search(question, top_k), output_type=RAGSearchResult)
    context_block = "\n\n".join(f"-{c}" for c in found.contexts)
    user_content =(
        "use the following context to interview the user like you are an expert hiring manager in this profession.\n\n"
        f"{context_block}\n\n"
        f"user question: {question}"
        "ask questions in a professional and serious manner"
    )

    adapter = ai.openai.Adapter(auth_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
    response = await ctx.step.ai.infer(
        "llm-answer",
        adapter = adapter,
        body = {
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You are an interviewer who only asks questions based on the context provided."},
                {"role": "user", "content": user_content}
            ]
        }
    )

    answer = response["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}
app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])



