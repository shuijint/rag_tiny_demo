from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fitness_main import inference
from milvus_fitness_crud import get_equipment_collection

# 创建FastAPI应用实例
app = FastAPI(
    title="示例API服务",
    description="一个演示FastAPI基本功能的示例服务",
    version="1.0.0",
)

model, tokenizer, device = None, None, None


# 定义数据模型
class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None


@app.post("/fitness_chat/")
def fitness_chat(key_name: str, question: str):
    collection = get_equipment_collection()
    result = inference(collection, key_name, question)
    return {"answer": result}


# 应用启动入口
if __name__ == "__main__":
    uvicorn.run(
        app="__main__:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        log_level="info",
        workers=1
    )
