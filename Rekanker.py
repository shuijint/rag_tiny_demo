from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# 1. 加载模型和分词器
model_name = "maidalun1020/bce-reranker-base_v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def reKanker(query, passages, TopK):
    pairs = [[query, passage] for passage in passages]
    with torch.no_grad():
        # 分词并转换为模型输入格式
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        # 模型推理
        outputs = model(**inputs)

        # 提取分数（使用sigmoid转换logits为概率）
        scores = torch.sigmoid(outputs.logits).squeeze(dim=1).numpy()

    # 5. 按分数降序排序文档
    ranked_results = [
        {"index": idx, "score": score, "text": passages[idx]}
        for idx, score in enumerate(scores)
    ]

    ranked_results.sort(key=lambda x: x["score"], reverse=True)

    # 打印排序结果
    print("查询内容:", query)
    print("\n排序结果:")
    context = "";
    for i, result in enumerate(ranked_results):
        print(f"Rank {i + 1} (Score: {result['score']:.4f}): {result['text']}")
        context = context + result['text'] + "\n"
        if i + 1 == TopK:
            break

    return context

