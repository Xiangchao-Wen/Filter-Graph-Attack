from ogb.nodeproppred import PygNodePropPredDataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import sparse_mx_to_torch_sparse_tensor
import torch
import numpy as np
from scipy.sparse import coo_matrix
import json

# # 加载 ogbn-arxiv 数据集
# dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/')
# data = dataset[0]
#
# # 转换为对 deeprobust 兼容的格式
# features = data.x.numpy()
# labels = data.y.numpy().squeeze()
# adj = coo_matrix((np.ones(data.edge_index.shape[1]),
#                   (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
#                  shape=(data.num_nodes, data.num_nodes))
#
# # 获取官方划分的训练、验证、测试集索引
# split_idx = dataset.get_idx_split()
# idx_train = split_idx['train']
# idx_val = split_idx['valid']
# idx_test = split_idx['test']
#
# # 转换为 PyTorch 张量
# features = torch.FloatTensor(features)
# labels = torch.LongTensor(labels)
# adj = sparse_mx_to_torch_sparse_tensor(adj)
#
# idx_train = torch.LongTensor(idx_train.numpy())
# idx_val = torch.LongTensor(idx_val.numpy())
# idx_test = torch.LongTensor(idx_test.numpy())
#
# # 定义和训练 GCN 模型
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# num_class = labels.max().item() + 1
#
# victim_model = GCN(nfeat=features.shape[1], nclass=num_class, nhid=256, dropout=0.5,
#                    weight_decay=5e-5, device=device).to(device)
#
# victim_model.fit(features.to(device), adj.to(device), labels.to(device), idx_train.to(device))
# victim_model.test(idx_test.to(device))

from utils import *
import argparse
from deeprobust.graph.global_attack import DICE
from deeprobust.graph.defense import GCN
from scipy.sparse import csr_matrix, lil_matrix
import time
from tqdm import tqdm
from scipy.sparse import coo_matrix, save_npz

device = 'cuda'

def extract_subgraph(data, dataset, ratio=0.1):
    """
    从原图中提取子图，包括邻接矩阵、特征矩阵、标签以及训练和测试集索引。

    参数:
        data: 原始数据对象，包含 edge_index, x, y 等属性。
        dataset: 数据集对象，包含 get_idx_split() 方法。
        ratio: 子图的节点占总节点数的比例 (默认值为 0.1)。

    返回:
        sub_adj: 子图的邻接矩阵 (稀疏格式)。
        sub_features: 子图的特征矩阵。
        sub_labels: 子图的标签数组。
        sub_idx_train: 子图的训练集索引。
        sub_idx_test: 子图的测试集索引。
    """
    # 总节点数
    num_nodes = data.num_nodes

    # 随机选择一定比例的节点
    num_sampled_nodes = int(ratio * num_nodes)
    sampled_nodes = np.random.choice(num_nodes, num_sampled_nodes, replace=False)

    # 创建节点索引映射表
    node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sampled_nodes)}

    # 构建原图的邻接矩阵
    adj = coo_matrix((np.ones(data.edge_index.shape[1]),
                      (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
                     shape=(num_nodes, num_nodes))

    # 确保矩阵对称性
    adj = (adj + adj.T)
    # 限制最大值为 1
    adj.data = np.clip(adj.data, 0, 1)
    # 转换为 CSR 格式以支持切片操作
    adj = adj.tocsr()
    # 提取子图邻接矩阵
    sub_adj = adj[sampled_nodes, :][:, sampled_nodes]

    # 提取子图对应的特征和标签
    sub_features = data.x.numpy()[sampled_nodes]
    sub_labels = data.y.numpy()[sampled_nodes]

    # 获取训练、验证、测试集的原始索引
    split_idx = dataset.get_idx_split()
    idx_train = split_idx['train'].numpy()
    idx_test = split_idx['test'].numpy()
    idx_val = split_idx['valid'].numpy()

    # print('ori_train_idx',idx_train)
    # 筛选并映射训练集和测试集索引
    sub_idx_train = [node_map[idx] for idx in idx_train if idx in node_map]
    sub_idx_test = [node_map[idx] for idx in idx_test if idx in node_map]
    sub_idx_val = [node_map[idx] for idx in idx_val if idx in node_map]
    # 转换为 PyTorch 张量
    sub_idx_train = torch.tensor(sub_idx_train, dtype=torch.long)
    sub_idx_test = torch.tensor(sub_idx_test, dtype=torch.long)
    sub_idx_val = torch.tensor(sub_idx_val, dtype=torch.long)

    # print('train_idx',sub_idx_train)
    # 确保邻接矩阵是稀疏格式
    sub_adj = csr_matrix(sub_adj)

    return sub_adj, sub_features, sub_labels, sub_idx_train, sub_idx_test, sub_idx_val

def get_new_adj(adj, adj_new, idx_list, batch_size=1000):
    # 将稀疏矩阵转换为 LIL 格式，便于修改
    # adj[idx_list, :] = adj_new[idx_list, :]
    # adj[:, idx_list] = adj_new[:, idx_list]

    adj = adj.tolil()  # 转换为 LIL 格式
    adj_new = adj_new.tolil()  # 如果需要，转换 adj_new 为 LIL 格式

    # 对于 idx_list 中的每个节点，逐行逐列地进行修改
    for i in tqdm(range(0, len(idx_list), batch_size), desc="Updating adjacency matrix", unit="batch"):
        batch_idx = idx_list[i:i + batch_size]  # 每次更新100个节点
        adj[batch_idx, :] = adj_new[batch_idx, :]
        # adj[:, batch_idx] = adj_new[:, batch_idx]

    # 确保矩阵对称性
    adj = (adj + adj.T)
    adj.data = np.clip(adj.data, 0, 1)

    # 返回转回 CSR 格式的稀疏矩阵
    return adj.tocsr()  # 转换回 CSR 格式进行后续处理
def filter_evasion(adj, mod_adj, features, idx_train, idx_val, idx_test, labels, repeat_n):
    incorrect_after_perturbation = set()
    for i in range(repeat_n):
        print(i)
        # clean graph
        gcn = GCN(nfeat=features.shape[1],
            nhid=16,
            nclass=labels.max().item() + 1,
            device='cuda')
        gcn = gcn.to('cuda')
        gcn.fit(features, adj, labels, idx_train, idx_val) # train on clean graph with earlystopping
        preds_before = gcn.predict(features, adj).argmax(1)
        # 在测试集上进行预测，并获取预测的概率分布
        preds_after = gcn.predict(features, mod_adj).argmax(1)
        # 保留在原始矩阵上预测正确的节点，并筛选出在扰动矩阵上预测错误的节点
        for idx in idx_test:  # 只考虑 idx_test 中的节点
            if preds_before[idx] != preds_after[idx]:
                incorrect_after_perturbation.add(idx)
        # 如果考虑，则保留过多的扰动，对过滤不利
        # for idx in idx_train:  # 考虑 idx_train 中的节点
        #     if preds_before[idx] != preds_after[idx]:
        #         incorrect_after_perturbation.add(idx)
    return list(incorrect_after_perturbation)  # 将 set 转换为列表并返回
def combine_deltas(deltas):
    # 确保 deltas 中所有矩阵形状一致
    num_matrices = len(deltas)
    num_rows, num_cols = deltas[0].shape
    assert all(delta.shape == (num_rows, num_cols) for delta in deltas), "矩阵形状不一致"

    # 初始化组合矩阵为 lil_matrix
    delta_combine = lil_matrix((num_rows, num_cols), dtype=np.float64)

    # 遍历每一行
    for i in range(num_rows):
        # 提取第 i 行的所有矩阵行
        rows = [delta.getrow(i).toarray().flatten() for delta in deltas]  # 转为 dense 行向量

        # 找到非全零行对应的矩阵索引
        non_zero_indices = [idx for idx, row in enumerate(rows) if np.any(row != 0)]

        if not non_zero_indices:  # 如果所有行都是全零
            continue
        else:
            # 从非全零行中筛选扰动最少的行
            non_zero_rows = [rows[idx] for idx in non_zero_indices]
            non_zero_counts = [np.count_nonzero(row) for row in non_zero_rows]
            min_index = non_zero_indices[np.argmin(non_zero_counts)]  # 对应的 delta 矩阵索引

            # 使用扰动最少的行更新 delta_combine
            delta_combine[i, :] = lil_matrix(rows[min_index])  # 更新行
            delta_combine[:, i] = lil_matrix(rows[min_index]).T  # 更新列（对称性）

    # 将矩阵中的元素截断为 [-1, 1]
    delta_combine = delta_combine.tocsr()  # 转换回 csr_matrix
    delta_combine.data = np.clip(delta_combine.data, -1, 1)

    return delta_combine


def combine_deltas_fast(deltas):

    """
    高效版本的 combine_deltas 函数，支持 SciPy 的 csr_matrix 格式，并使用 PyTorch 加速。

    Args:
        deltas (list of csr_matrix): 输入的稀疏矩阵列表，所有矩阵形状一致。

    Returns:
        csr_matrix: 合并后的稀疏矩阵。
    """
    # 检查所有矩阵形状一致
    num_matrices = len(deltas)
    num_rows, num_cols = deltas[0].shape
    assert all(delta.shape == (num_rows, num_cols) for delta in deltas), "矩阵形状不一致"

    # 将 deltas 转换为 PyTorch 稠密张量并移到 GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deltas_dense = [torch.tensor(delta.toarray(), device=device) for delta in deltas]

    # 堆叠所有矩阵形成 3D 张量 (num_matrices, num_rows, num_cols)
    stacked = torch.stack(deltas_dense, dim=0)

    # 计算每个位置的非零元素数目
    non_zero_mask = stacked != 0  # 每个元素是否非零
    non_zero_count = non_zero_mask.sum(dim=0)  # 每个位置的非零元素数

    # 找到扰动最少的矩阵索引
    # 确保 min_count 的类型为 float，以支持 inf 值
    min_count = torch.where(
        non_zero_count > 0,
        non_zero_count.float(),  # 将 non_zero_count 转为 float
        torch.full_like(non_zero_count, float('inf'), dtype=torch.float)  # 明确指定 dtype
    )
    min_indices = torch.argmin(min_count, dim=0)  # 对于每个位置，选择非零元素最少的矩阵索引

    # 根据 min_indices 构建 delta_combine
    row_indices, col_indices = torch.meshgrid(
        torch.arange(num_rows, device=device),
        torch.arange(num_cols, device=device),
        indexing="ij"
    )
    selected_delta = stacked[min_indices, row_indices, col_indices]

    # 将选出的矩阵裁剪到 [-1, 1]
    delta_combine = torch.clamp(selected_delta, -1, 1)

    # 转换回 SciPy 稀疏矩阵
    delta_combine_numpy = delta_combine.cpu().numpy()  # 移回 CPU 并转换为 NumPy
    delta_combine_sparse = csr_matrix(delta_combine_numpy)  # 转为 SciPy 的 csr_matrix
    return delta_combine_sparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptb_rate', type=float, default=0.5)
    parser.add_argument('--repeat_n', type=int, default=1)
    parser.add_argument('--select_times', type=int, default=1)
    parser.add_argument('--combine_n', type=int, default=1)

    # parser.add_argument('--filter_n', type=int, default=20)

    args = parser.parse_args()
    repeat_n = args.repeat_n
    combine_n = args.combine_n
    select_times = args.select_times
    device = 'cuda'

    logger_filename = 'DICE-'+'large' + '.log'
    logger_name = 'evaluate'
    logger = get_logger(logger_filename, level=0, name=logger_name)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/')
    data = dataset[0]
    print(data.edge_index.size(1))
    # 转换为对 deeprobust 兼容的格式
    # features = data.x.numpy()
    # labels = data.y.numpy().squeeze()
    # print(labels)
    # adj = coo_matrix((np.ones(data.edge_index.shape[1]),
    #                   (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
    #                  shape=(data.num_nodes, data.num_nodes))
    # adj = (adj + adj.T).multiply(0.5)
    #
    # # 获取官方划分的训练、验证、测试集索引
    # split_idx = dataset.get_idx_split()
    # idx_train = split_idx['train']
    # idx_val = split_idx['valid']
    # idx_test = split_idx['test']

    results = []
    for i in range(1):

        adj, features, labels, idx_train, idx_test, idx_val = load_data('ogb','arxiv')
        # adj, features, labels, idx_train, idx_test, idx_val = extract_subgraph(data, dataset, ratio=0.1)
        # adj_copy = adj.copy()

        labels = labels.squeeze()

        # 转换为 PyTorch 张量

        n_perturbations = int(args.ptb_rate * (adj.nnz // 2))
        num_class = labels.max().item() + 1

        victim_model = GCN(nfeat=features.shape[1], nclass=num_class, nhid=256, dropout=0.5,
                           weight_decay=5e-5, device=device).to(device)

        print("训练前邻接矩阵的最大值:", adj.data.max())
        victim_model.fit(features, adj, labels, idx_train)
        print("训练后邻接矩阵的最大值:", adj.data.max())

        victim_model.test(idx_test)
        print("训练后邻接矩阵的最大值:", adj.data.max())

        # 调用 predict 时，确保输入数据也在同一设备
        predicted_labels = victim_model.predict(features, adj).argmax(1)
        print("训练后邻接矩阵的最大值:", adj.data.max())
        # 假设 labels 是 numpy.ndarray，predicted_labels 是 Tensor
        predicted_labels_numpy = predicted_labels.cpu().numpy()  # 将 tensor 转换为 numpy.ndarray
        predicted_labels_tensor = torch.LongTensor(predicted_labels_numpy).to(device)

        pseudo_labels = labels.copy()  # 使用 numpy 的 copy 方法来克隆 labels
        # 使用 numpy 的 isin 函数进行布尔索引替换
        pseudo_labels[~np.isin(np.arange(labels.shape[0]), idx_train)] = predicted_labels_numpy[
            ~np.isin(np.arange(labels.shape[0]), idx_train)]

        start_time = time.time()
        model = DICE()
        deltas = []
        for _ in range(combine_n):
            modified_adj = adj.tolil()
            print('max ori adj', modified_adj.tocsr().max())
            model.attack(adj, pseudo_labels, n_perturbations=n_perturbations)
            # modified_adj = model.modified_adj
            mod_adj = sp.csr_matrix(model.modified_adj)
            print('attack_done')
            selected_idx_test = filter_evasion(adj, mod_adj, features, idx_train, idx_val, idx_test, labels, select_times)
            # selected_idx_test = filter_evasion_pyg(pyg_data, mod_adj, idx_test, select_times, victim_pyg)
            final_filter_adj = get_new_adj(adj, mod_adj, selected_idx_test)
            preds_evasion = victim_model.predict(features, final_filter_adj).argmax(1)
            attack_acc_filter = calculate_accuracy(idx_test, preds_evasion, labels)
            print('filter_acc', attack_acc_filter)
            filter_changed_edges = change_num(adj, final_filter_adj)
            # 过滤扰动数量
            total_filter_edges = n_perturbations - filter_changed_edges
            print('filter_n', total_filter_edges, filter_changed_edges)
            filename='./attack_data/ogb/DICE-arxiv-0.1.npz'
            save_npz(filename, adj)  # 保存稀疏矩阵
            # delta = final_filter_adj - adj
            # print('delta',delta.nnz/2)
            # deltas.append(delta)
            print('attack_rate',filter_changed_edges/(adj.nnz // 2))

            model.attack(adj, pseudo_labels, n_perturbations=filter_changed_edges)
            mod_adj = sp.csr_matrix(model.modified_adj)
            preds_evasion = victim_model.predict(features, mod_adj).argmax(1)
            attack_acc_DICE = calculate_accuracy(idx_test, preds_evasion, labels)
            print('DICE_acc', attack_acc_DICE)


        # delta_combine = combine_deltas(deltas)
        # combine_DICE = adj + delta_combine
        # selected_idx_test = filter_evasion_pyg(pyg_data, combine_DICE, idx_test, select_times, victim_pyg)
        # combine_filter_adj = get_new_adj(adj, combine_DICE, selected_idx_test)
        #
        # changed_edges = change_num(adj, combine_filter_adj)
        #
        # mod_adj_pyg = retype_adj(combine_filter_adj)
        # victim_pyg = load_pyg_model(pyg_data, args.victim, args.source, args.dataset, device, logger, True)
        # preds_evasion = victim_pyg.predict(pyg_data.x, mod_adj_pyg).argmax(1)
        # attack_acc = calculate_accuracy(idx_test, preds_evasion, labels)
        # print('combine changes', changed_edges,attack_acc)
        end_time = time.time()
        print(f"Time taken to combine deltas: {end_time - start_time:.6f} seconds")
        results.append([0.02*(i+1), end_time - start_time, filter_changed_edges, filter_changed_edges/(adj.nnz // 2), attack_acc_filter, attack_acc_DICE])

    columns = ['rate', 'time', 'attack_n', 'attack_rate', 'filter_acc', 'DICE_acc']
    results_df = pd.DataFrame(results, columns=columns)

    # 打印 DataFrame 预览
    print(results_df)

    # 将 DataFrame 保存为 CSV 文件
    results_df.to_csv('results_products.csv', index=False)

    # # 画图
    # plt.figure(figsize=(8, 6))
    # plt.plot([0.1 * (i + 1) for i in range(len(times))], times, marker='o', linestyle='-', color='b')
    # plt.title("Time Taken for Different Ratios of Nodes", fontsize=14)
    # plt.xlabel("Ratio of Nodes", fontsize=12)
    # plt.ylabel("Time (seconds)", fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.savefig('time_plot.pdf',format='pdf', bbox_inches="tight")  # 保存图像到文件
    # plt.show()
