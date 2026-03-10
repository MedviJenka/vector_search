import pytest
import random
from typing import List
from dataclasses import dataclass
from functools import cached_property
from faker import Faker
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

from settings import Config


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
            vectors_config=VectorParams(
                size=self.config.size,
                distance=Distance.COSINE
            )
        )

    def insert_vectors(
        self,
        vectors: List[List[float]],
        ids: List[int],
        payloads: List[dict]
    ) -> UpdateResult:

        return self.client.upsert(
            collection_name=self.config.collection_name,
            points=Batch(
                ids=ids,
                vectors=vectors,
                payloads=payloads
            )
        )

    def retrieve_vectors(self, ids: List[int]) -> List[Record]:
        return self.client.retrieve(
            collection_name=self.config.collection_name,
            ids=ids,
            with_vectors=True
        )

    @staticmethod
    def build_filter(key: str, value: str) -> Filter:
        return Filter(
            must=[FieldCondition(key=key, match=MatchValue(value=value))]
        )


# ---------------- #
#   Fake Dataset   #
# ---------------- #

fake = Faker()


@dataclass
class FakeVectorDataset:
    ids: List[int]
    vectors: List[List[float]]
    payloads: List[dict]


def generate_fake_dataset(count: int, vector_size: int) -> FakeVectorDataset:

    ids = list(range(count))

    vectors = [
        [random.random() for _ in range(vector_size)]
        for _ in range(count)
    ]

    payloads = [
        {
            "name": fake.name(),
            "address": fake.address(),
            "url": fake.url(),
            "year": fake.year(),
            "country": fake.country(),
            "email": fake.email(),
            "company": fake.company(),
            "job": fake.job(),
        }
        for _ in range(count)
    ]

    return FakeVectorDataset(
        ids=ids,
        vectors=vectors,
        payloads=payloads
    )


# ---------------- #
#   Test Fixture   #
# ---------------- #

@pytest.fixture(scope="module")
def vector_model():

    config = VectorConfig(
        collection_name="test_collection1",
        size=128
    )

    model = VectorModel(config)

    try:
        model.client.delete_collection(config.collection_name)
    except Exception:
        pass

    model.create_collection()

    return model


# ---------------- #
#       Tests      #
# ---------------- #

class Test:

    def test_create_collection(self, vector_model: VectorModel):

        collections = vector_model.client.get_collections()
        names = [c.name for c in collections.collections]

        assert vector_model.config.collection_name in names

    def test_insert_vectors(self, vector_model: VectorModel):

        dataset = generate_fake_dataset(
            count=10,
            vector_size=vector_model.config.size
        )

        result = vector_model.insert_vectors(
            vectors=dataset.vectors,
            ids=dataset.ids,
            payloads=dataset.payloads
        )

        assert result.status.name == "COMPLETED"

    def test_retrieve_vectors(self, vector_model: VectorModel):

        dataset = generate_fake_dataset(
            count=10,
            vector_size=vector_model.config.size
        )

        vector_model.insert_vectors(
            vectors=dataset.vectors,
            ids=dataset.ids,
            payloads=dataset.payloads
        )

        records = vector_model.retrieve_vectors(dataset.ids)

        assert len(records) == len(dataset.ids)
        assert isinstance(records[0], Record)

    @pytest.mark.parametrize("country", ["Israel", "Australia", "England", "Germany"])
    def test_build_filter(self, country: str):

        f = VectorModel.build_filter("country", country)
        print(f)

        assert isinstance(f, Filter)
        assert f.must[0].key == "country"
        assert f.must[0].match.value == country
