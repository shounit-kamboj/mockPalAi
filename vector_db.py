from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=3072):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(ids, vectors, payloads)]
        self.client.upsert(self.collection, points=points)

    def _query(self, query_vector, top_k: int):
        return self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        ).points

    # Old PDF search; delete with rag_query_pdf_ai in step 6.
    def search(self, query_vector, top_k: int = 5):
        contexts = []
        sources = set()
        for point in self._query(query_vector, top_k):
            payload = point.payload or {}
            text = payload.get("text", "")
            if text:
                contexts.append(text)
                sources.add(payload.get("source", ""))
        return {"contexts": contexts, "sources": list(sources)}

    # Returns each hit's full payload plus its similarity score (higher = closer).
    def search_questions(self, query_vector, top_k: int = 5) -> list[dict]:
        return [{**(point.payload or {}), "score": point.score} for point in self._query(query_vector, top_k)]