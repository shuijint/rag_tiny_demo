import requests


def ollama_embedding_by_api(text):
    res = requests.post(
        url="http://127.0.0.1:11434/api/embeddings",
        json={
            "model": "nomic-embed-text:latest",
            "prompt": text
        }
    )
    # print(res.json())
    embedding = res.json()['embedding']
    return embedding


def ollama_generate_by_api(prompt):
    response = requests.post(
        url="http://127.0.0.1:11434/api/generate",
        json={
            "model": "qwen3:4b",
            "prompt": prompt,
            "stream": False,
            'temperature': 0
        }
    )
    res = response.json()['response']
    return res



