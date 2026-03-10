from typing import List
from settings import Config
from dataclasses import dataclass
from functools import cached_property
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Batch,
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    UpdateResult,
    Record
)


# ---------------- #
#   Vector Model   #
# ---------------- #

@dataclass
class VectorModel:

    config: Config

    @cached_property
    def client(self) -> QdrantClient:
        return QdrantClient(host="localhost", port=6333)

    def create_collection(self) -> None:
        self.client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(size=self.config.size, distance=Distance.COSINE))

    def insert_vectors(self, vectors: List[List[float]], ids: List[int], payloads: List[dict]) -> UpdateResult:
        return self.client.upsert(collection_name=self.config.collection_name, points=Batch(ids=ids, vectors=vectors, payloads=payloads))

    def retrieve_vectors(self, ids: List[int]) -> List[Record]:
        return self.client.retrieve(collection_name=self.config.collection_name, ids=ids, with_vectors=True)

    @staticmethod
    def build_filter(key: str, value: str) -> Filter:
        return Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))])


v = VectorModel(config=Config(collection_name='Main2'))
v.create_collection()
