from LLm import ollama_embedding_by_api, ollama_generate_by_api
from milvus_fitness_crud import get_equipment_collection, add_data, combined_search
from Rekanker import reKanker


def inference(collection, equip_name, question):
    qs_embedding = ollama_embedding_by_api(question)
    search_result = combined_search(collection, equip_name, qs_embedding)
    contextList = []
    for content in search_result:
        contextList.append(content["equip_content"])
    # rekanker 检索增强操作
    context = reKanker(question, contextList,1)
    prompt = f"""你是一个健身问答的私人机器人，任务是根据参考信息回答用户问题，如果参考信息不足以回答用户问题或者没有关系，请回复不知道，不要去杜撰任何信息！！！请用中文回答。
    参考信息：{context}，来回答问题：{question}，
    """
    result = ollama_generate_by_api(prompt)
    return result


if __name__ == '__main__':
    collection = get_equipment_collection()
    # equip_content = "单杠的操作方法：做垂直向上的引体向上，力量不足者可使用较低的单杠做斜向引体向上。"
    # embedding = ollama_embedding_by_api(equip_content)
    # add_result = add_data(collection,"单杠的操作方法",embedding,equip_content)
    # print(add_result)
    equip_name = "杠铃"
    question = "请问杠铃的动作要领是什么？"
    result = inference(collection, equip_name, question)
    print("=========结果===========")
    print(result)
