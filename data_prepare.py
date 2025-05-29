import os
import pickle
from collections import defaultdict
from utils.util import find_all_json
import json
import random

dataset_name = "IED"
graph_size = 8  # 8/12/16
split_count = [5000, 1500, 1000]
candidate_count = 12

print(f'预处理：在{dataset_name}数据集上提取大小为{graph_size}的固定子图及其时序增强图')

if dataset_name == "IED":
    # 获取 IED
    data_train = find_all_json("./data/origin/Wiki_IED_split/train")
    data_test = find_all_json("./data/origin/Wiki_IED_split/test")
    data_dev = find_all_json("./data/origin/Wiki_IED_split/dev")
else:
    # 获取 LDC
    data_train = find_all_json("./data/origin/LDC_schema_corpus_ce_split/train")
    data_test = find_all_json("./data/origin/LDC_schema_corpus_ce_split/test")
    data_dev = find_all_json("./data/origin/LDC_schema_corpus_ce_split/dev")

all_data = [item for items in [data_train, data_test, data_dev] for item in items]

print(f'共有{len(all_data)}条数据')


# ================== 新增全局概率计算 ==================
def compute_global_probabilities(all_effe_paths):
    """
    全局统计事件转移概率 p(e_a | e_b, t)
    返回结构：{(e_b_type, t): {e_a_type: count}}
    """
    # 统计 (e_b, t) 条件下 e_a 的出现次数
    prob_counts = defaultdict(lambda: defaultdict(int))

    for path in all_effe_paths:
        # 将路径中的索引转换为事件类型
        event_types = path

        # 遍历所有可能的事件对
        for i in range(len(event_types)):
            for j in range(i + 1, min(len(event_types), i + 4)):
                e_a = event_types[j]
                e_b = event_types[i]
                t = j - i  # 时间差

                prob_counts[(e_b, t)][e_a] += 1

    # 转换为概率
    global_probs = defaultdict(dict)
    for (e_b, t), counts in prob_counts.items():
        total = sum(counts.values())
        for e_a, cnt in counts.items():
            global_probs[(e_b, t)][e_a] = cnt / total if total > 0 else 0.0

    return global_probs


# 深度遍历图
def edges_to_adjacency_list(edges):
    adjacency_list = {}
    for edge in edges:
        head, tail = edge  # 假设每个表示是一个包含头节点和尾节点的元组
        if head in adjacency_list:
            adjacency_list[head].append(tail)
        else:
            adjacency_list[head] = [tail]
    return adjacency_list


# 生成所有有效路径（原有DFS逻辑）
def dfs(node, path, visited, adjacency_list, depth=0):
    path.append(node)
    visited.add(node)

    if depth > graph_size:
        return

    # 如果当前节点是终点（没有出边），则将路径添加到路径列表中
    if node not in adjacency_list:
        yield path.copy()

    # 继续遍历后续节点
    for neighbor in adjacency_list.get(node, []):
        if neighbor not in visited:
            yield from dfs(neighbor, path, visited, adjacency_list, depth + 1)

    path.pop()
    visited.remove(node)


# ================== 改进的create_event_graph ==================
def create_event_graph(graph):
    """返回：事件列表、边列表、有效路径（用于后续处理）"""
    events = []
    # 抽取事件信息
    for event in graph["schemas"][0]["steps"]:
        event["@type"] = event["@type"].split("/")[-1]
        events.append(event)

    # 构建边列表
    t_list = []
    for t in graph["schemas"][0]["order"]:
        source, target = None, None
        for e_id, e in enumerate(events):
            if t["before"] == e["@id"]:
                source = e_id
            if t["after"] == e["@id"]:
                target = e_id
        if source is not None and target is not None:
            t_list.append((source, target))

    adjacency_list = edges_to_adjacency_list(t_list)
    all_paths = []
    max_path = []
    for node in adjacency_list:
        for path in dfs(node, [], set(), adjacency_list):
            if len(path) > len(max_path):
                max_path = path
    all_paths.append(max_path)

    events = [event["@type"] for event in events]

    return events, t_list, all_paths


all_effe_paths = []

test_list = []
dev_list = []
train_list = []

saved_data = []

# 第一次遍历：收集所有路径
for data in all_data:
    events, t_list, paths = create_event_graph(data)
    saved_data.append([events, t_list])
    all_effe_paths.extend([[events[idx] for idx in path] for path in paths])

print(f'所有路径收集完毕')

# 计算全局概率
global_probs = compute_global_probabilities(all_effe_paths)

print(f'全局概率计算完毕')


def generate_subgraphs(events, adjacency_list, graph_size, global_probs=None):
    """生成固定大小子图，及时序增强后的邻接表"""
    subgraphs = []

    # 随机选择 graph_size 的索引
    events_index = list(range(len(events)))
    n = len(events) - graph_size - 1
    if n <= 0:
        return subgraphs
    t_set = []
    for i in range(n):
        for _ in range(graph_size // 2):
            sub_path = sorted(list(random.sample(events_index[i: graph_size * 2 + i], graph_size)))
            t_str = ",".join([str(tt) for tt in sub_path])
            if t_str in t_set:
                continue
            t_set.append(t_str)
            # sub_path = list(range(i, i + graph_size))
            sub_events = [events[idx] for idx in sub_path]

            # --- 构建邻接表（包含虚拟节点）---
            # 原始边（调整索引为子图局部索引）
            sub_edges = []
            for u in range(len(sub_path)):
                original_u = sub_path[u]
                for v in adjacency_list.get(original_u, []):
                    if v in sub_path:
                        v_sub = sub_path.index(v)
                        sub_edges.append((u, v_sub))

            # 节点过少不考虑
            if len(sub_edges) < graph_size - 1:
                continue

            # 添加虚拟节点逻辑
            # start节点（索引为-1）连接无前驱的节点
            # end节点（索引为graph_size）连接无后续的节点
            has_incoming = {v for _, v in sub_edges}
            for u in range(len(sub_path)):
                if u not in has_incoming:
                    sub_edges.append((-1, u))  # start -> u
            has_outgoing = {u for u, _ in sub_edges}
            for v in range(len(sub_path)):
                if v not in has_outgoing:
                    sub_edges.append((v, graph_size))  # v -> end

            enhanced_edges = [[items[0], items[1], 1.0] for items in sub_edges]
            # --- 时序增强：基于全局概率添加边 ---
            if global_probs is not None:
                for u in range(len(sub_path)):
                    e_b = sub_events[u]
                    for t in [1, 2]:  # 假设只考虑1/2步转移
                        if (e_b, t) in global_probs:
                            for e_a, prob in global_probs[(e_b, t)].items():
                                if (u + t) < len(sub_path):
                                    e_a_actual = sub_events[u + t]
                                    if e_a == e_a_actual and (u, u + t) not in enhanced_edges:
                                        enhanced_edges.append([u, u + t, prob])

            # 真值：下一个事件（若存在）
            target = None
            if sub_path[len(sub_path) - 1] + 1 < len(events):
                target_event = events[sub_path[-1] + 1]
                target = target_event

            if target is None:
                continue

            candidate = list(select_n_without_target([k for k in type2id], target, candidate_count - 1))
            insert_index = random.randint(0, candidate_count - 1)
            candidate.insert(insert_index, target)

            # 添加起始节点和结束节点
            sub_events.insert(0, "start")
            sub_events.append("end")
            sub_edges = [[item + 1 for item in items] for items in sub_edges]
            enhanced_edges = [[items[0] + 1, items[1] + 1, items[2]] for items in enhanced_edges]

            subgraphs.append({
                "events": sub_events,
                "edges": sub_edges,
                "enhanced_edges": enhanced_edges,
                "target": target,
                "target_index": insert_index,
                "candidate": candidate
            })

    return subgraphs


def select_n_without_target(original_list, target, n):
    # 过滤掉目标元素
    filtered_list = [x for x in original_list if x != target]
    # 检查过滤后的列表长度是否足够
    if len(filtered_list) < n:
        raise ValueError("过滤后的列表元素不足四个，无法进行选择")
    # 随机选择四个元素
    return random.sample(filtered_list, n)


with open("./data/prepared/ontology_embeddings.pkl", "rb") as f:
    ontology_embeddings = pickle.load(f)
ontology, type2id, type_embeddings = ontology_embeddings

# 修改第二次遍历代码
result = []
data_sources = {  # 记录数据原始来源
    "train": data_train,
    "test": data_test,
    "dev": data_dev
}

print(f'数据预处理中。。。')

for index, [events, t_list] in enumerate(saved_data):
    print(f'第{index + 1}批数据开始处理')

    adjacency_list = edges_to_adjacency_list(t_list)

    # 生成子图（包含时序增强）
    subgraphs = generate_subgraphs(events, adjacency_list, graph_size, global_probs)

    result.extend(subgraphs)
    print(f'第{index + 1}批数据处理完毕')
    if len(result) >= sum(split_count):
        break

print(f'共处理出{len(result)}条数据')

train_list.extend(result[0: split_count[0]])
test_list.extend(result[split_count[0]: split_count[0] + split_count[1]])
dev_list.extend(result[split_count[0] + split_count[1]:])

# 确保输出目录存在
output_dirs = [
    f'./data/prepared/{dataset_name}/test',
    f'./data/prepared/{dataset_name}/train',
    f'./data/prepared/{dataset_name}/dev'
]
for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)

print(f'开始写入文件')

# 写入文件（添加ensure_ascii=False避免中文乱码）
with open(f'./data/prepared/{dataset_name}/test/node{graph_size}.json', 'w') as f:
    json.dump(test_list, f, ensure_ascii=False, indent=2)
with open(f'./data/prepared/{dataset_name}/train/node{graph_size}.json', 'w') as f:
    json.dump(train_list, f, ensure_ascii=False, indent=2)
with open(f'./data/prepared/{dataset_name}/dev/node{graph_size}.json', 'w') as f:
    json.dump(dev_list, f, ensure_ascii=False, indent=2)

print(f'写入完成！！！')
