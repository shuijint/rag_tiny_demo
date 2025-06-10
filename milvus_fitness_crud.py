import random
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility


# 连接到 Milvus 服务
def get_equipment_collection():
    # 连接到 Milvus 服务
    try:
        connections.connect(
            alias="default",
            host='172.24.81.205',
            port='19530'
        )
        print("成功连接到 Milvus 服务")
        equipment_collection = Collection("fitness_equipment")
        equipment_collection.load()
        print("成功获取并加载 fitness_equipment collection")
        return equipment_collection
    except Exception as e:
        print(f"获取 collection 失败: {e}")
        return None


# 添加数据
def add_data(collection, equip_name, equip_vector,equip_content):

    data = [
        # [id],
        [equip_vector],
        [equip_name],
        [equip_content]
    ]

    # 插入数据
    insert_result = collection.insert(data)
    print(f"成功插入数据，ID: {insert_result.primary_keys}")
    return insert_result.primary_keys[0]


def update_data(collection, equip_name, equip_vector,equip_content, id):
    try:
        # 1. 删除旧记录
        delete_expr = f"id == {id}"
        delete_result = collection.delete(delete_expr)
        print(f"已删除 ID 为 {id} 的记录，影响行数: {delete_result}")

        data = [
            [id],  # 保持相同的 ID
            [equip_vector],  # 向量字段
            [equip_name], # 设备名称
            [equip_content],
        ]

        # 3. 插入新记录
        insert_result = collection.insert(data)
        new_id = insert_result.primary_keys[0]
        print(f"成功更新数据，新记录 ID: {new_id}")

        # 4. 刷新 collection 以确保变更生效
        collection.flush()

        return new_id
    except Exception as e:
        print(f"更新失败: {e}")
        return None


def delete_data(collection, id):
    try:
        delete_expr = f"id == {id}"
        delete_result = collection.delete(delete_expr)
        print(f"已删除 ID 为 {id} 的记录，影响行数: {delete_result}")
        collection.flush()
        return delete_result
    except Exception as e:
        print(f"删除失败: {e}")
        return None


def search_by_name(collection, name_keyword, limit=10):
    """根据设备名称关键字模糊查询"""
    try:
        # 使用 LIKE 进行模糊匹配
        expr = f"equip_name LIKE \"%{name_keyword}%\""

        results = collection.query(
            expr=expr,
            output_fields=["id", "equip_name", "equip_vector"],
            limit=limit
        )

        # print(f"找到 {len(results)} 条匹配记录")
        return results
    except Exception as e:
        print(f"查询失败: {e}")
        return []


# 组合查询（名称＋向量）
def combined_search(collection, name_keyword, query_vector, limit=10):
    """组合名称关键词和向量相似度进行搜索"""
    try:
        # 先进行名称过滤
        name_results = search_by_name(collection, name_keyword, limit=10)

        if not name_results:
            print("没有匹配名称的记录")
            return []

        # 获取匹配名称的记录 ID
        ids = [result["id"] for result in name_results]

        # 构建 ID 过滤表达式
        id_expr = " || ".join([f"id == {id}" for id in ids])

        # 向量搜索并应用 ID 过滤
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }

        results = collection.search(
            data=[query_vector],
            anns_field="equip_vector",
            param=search_params,
            expr=id_expr,  # 应用 ID 过滤
            limit=limit,
            output_fields=["id", "equip_name", "equip_vector","equip_content"]
        )

        hits = results[0]
        # print(f"找到 {len(hits)} 条组合匹配记录")

        results_list = []
        for hit in hits:
            results_list.append({
                # "id": hit.id,
                # "equip_name": hit.entity.get("equip_name"),
                # "distance": hit.distance,
                # "equip_vector": hit.entity.get("equip_vector")
                "equip_content": hit.entity.get("equip_content")
            })

        return results_list
    except Exception as e:
        print(f"组合搜索失败: {e}")
        return []


