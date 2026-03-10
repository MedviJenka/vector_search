import pytest
import random
from typing import List
from dataclasses import dataclass
from faker import Faker
from main import VectorModel, VectorConfig
from qdrant_client.http.models import Filter, Record


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
    config = VectorConfig(collection_name="test_collection1", size=128)
    model = VectorModel(config)

    try:
        model.client.delete_collection(config.collection_name)
    except Exception as e:
        raise e

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
