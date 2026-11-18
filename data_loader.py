from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

#chunk the pdf sentences data with overlap

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=512, chunk_overlap=75)

def load_and_chunk_pdf(file_path: str):
    docs = PDFReader().load_data(file = file_path)
    texts = [doc.text for doc in docs if getattr(doc, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    for i, c in enumerate(chunks):
        print(f"--- chunk {i} ({len(c)} chars) ---")
        print(c)
        print()
    return chunks

def embed_text(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]



