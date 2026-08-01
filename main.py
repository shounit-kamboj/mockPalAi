import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime



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
    return{"hello": "world"}

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf])



