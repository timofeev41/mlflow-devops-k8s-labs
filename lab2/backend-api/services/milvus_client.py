from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import os

class MilvusService:
    """Клиент для взаимодействия с Milvus."""
    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        connections.connect(host=self.host, port=self.port)

    @staticmethod
    def create_collection(name="demo_collection", dim=128):
        if utility.has_collection(name):
            return Collection(name)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="payload", dtype=DataType.JSON)
        ]
        schema = CollectionSchema(fields, description="Demo collection")
        return Collection(name=name, schema=schema)

    def insert(self, name, ids, vectors, payloads):
        collection = self.create_collection(name)
        entities = [ids, vectors, payloads]
        collection.insert(entities)

    @staticmethod
    def search(name, query_vector, top_k=5):
        collection = Collection(name)
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "L2"},
            limit=top_k,
            output_fields=["id", "payload"]
        )
        return results[0]
