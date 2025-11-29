import logging
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import json
import os
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import redis

# Setup logging
logging.basicConfig(level=logging.INFO)

# Vector store Configuration
MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
MILVUS_PORT = int(os.getenv('MILVUS_PORT', '19530'))
COLLECTION_NAME = "demo_collection"
VECTOR_SIZE = 128

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=6379,
    db=0,
)

connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

def ensure_collection(name=COLLECTION_NAME, dim=VECTOR_SIZE):
    if utility.has_collection(name):
        return Collection(name)
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="payload", dtype=DataType.JSON)
    ]
    schema = CollectionSchema(fields, description="Demo collection")
    return Collection(name, schema=schema)

milvus_collection = ensure_collection()

app = FastAPI(title="FastAPI + Redis + Milvus Demo")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    """Отрисовать фронтенд."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/add")
def add_vector_form(
    request: Request,
    vector_id: int = Form(...),
    vector_data: str = Form(...),
    payload_data: str = Form("{}")
):
    """Добавить вектор через форму."""
    try:
        vector = [float(x.strip()) for x in vector_data.split(",")]
        if len(vector) != VECTOR_SIZE:
            vector = (vector + [0.0] * VECTOR_SIZE)[:VECTOR_SIZE]
        payload = json.loads(payload_data)

        entities = [
            [vector_id],
            [vector],
            [payload]
        ]
        milvus_collection.insert(entities)
        redis_client.set(f"vector:{vector_id}", json.dumps(vector), ex=300)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": f"Vector {vector_id} added successfully!"
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Error: {str(e)}"
        })

@app.get("/vector/{vector_id}")
def get_vector(vector_id: int):
    """Получить вектор по ID, сначала из Redis (если есть в кеше), затем из Milvus."""
    # Сначала ищем в Redis
    cached = redis_client.get(f"vector:{vector_id}")
    if cached:
        logging.info(f"Vector {vector_id} found in Redis cache")
        return {"id": vector_id, "vector": json.loads(cached), "source": "cache"}

    expr = f"id == {vector_id}"
    results = milvus_collection.query(expr, output_fields=["id", "vector", "payload"])
    if results:
        vector = results[0]["vector"]
        payload = results[0]["payload"]
        redis_client.set(f"vector:{vector_id}", json.dumps(vector), ex=300)
        return {"id": vector_id, "vector": vector, "payload": payload, "source": "milvus"}
    raise HTTPException(status_code=404, detail="Vector not found")

@app.post("/search")
def search_vectors_form(
    request: Request,
    query_vector: str = Form(...),
    top_k: int = Form(5)
):
    """Поиск похожих векторов через форму."""
    try:
        query = [float(x.strip()) for x in query_vector.split(",")]
        if len(query) != VECTOR_SIZE:
            query = (query + [0.0] * VECTOR_SIZE)[:VECTOR_SIZE]
        results = milvus_collection.search(
            data=[query],
            anns_field="vector",
            param={"metric_type": "L2"},
            limit=top_k,
            output_fields=["id", "payload"]
        )[0]
        found = []
        for r in results:
            found.append({
                "id": r.id,
                "score": r.distance,
                "payload": r.payload
            })
        return templates.TemplateResponse("index.html", {
            "request": request,
            "search_results": found,
            "query": query_vector
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Search error: {str(e)}"
        })

@app.get("/health")
def health_check():
    """Проверка состояния сервисов (healthcheck)."""
    health = {"status": "ok", "services": {}}
    # Redis
    try:
        if redis_client.ping():
            health["services"]["redis"] = "ok"
        else:
            health["services"]["redis"] = "error"
    except Exception as e:
        health["services"]["redis"] = f"error: {str(e)}"
    # Milvus
    try:
        if utility.has_collection(COLLECTION_NAME):
            health["services"]["milvus"] = "ok"
        else:
            health["services"]["milvus"] = "no_collection"
    except Exception as e:
        health["services"]["milvus"] = f"error: {str(e)}"
    if all(val == "ok" for val in health["services"].values()):
        health["status"] = "ok"
    else:
        health["status"] = "degraded"
    return health

if __name__ == "__main__":
    # entrypoint для запуска аппки
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
