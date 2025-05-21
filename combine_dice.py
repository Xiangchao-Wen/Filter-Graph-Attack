from utils import *
import argparse
from deeprobust.graph.global_attack import DICE
from deeprobust.graph.defense import GCN
from scipy.sparse import csr_matrix, lil_matrix
import time

device = 'cuda'
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
    parser.add_argument('--dataset', type=str, default='blogcatalog',
                        choices=['cora', 'citeseer', 'acm', 'cora_ml', 'polblogs','blogcatalog'])
    parser.add_argument('--victim', type=str, default='gcn', choices=['gcn', 'sgc', 'gat', 'graph-sage'])
    parser.add_argument('--source', type=str, default='gcn', choices=['planepoied', 'gcn', 'prognn'])
    parser.add_argument('--ptb_rate', type=float, default=0.5)
    parser.add_argument('--repeat_n', type=int, default=1)
    parser.add_argument('--select_times', type=int, default=1)
    parser.add_argument('--combine_n', type=int, default=10)
    parser.add_argument('--pretrained', type=bool, default=True)

    # parser.add_argument('--filter_n', type=int, default=20)

    args = parser.parse_args()
    repeat_n = args.repeat_n
    combine_n = args.combine_n
    select_times = args.select_times
    device = 'cuda'

    logger_filename = 'DICE-' + args.dataset + '.log'
    logger_name = 'evaluate'
    logger = get_logger(logger_filename, level=0, name=logger_name)

    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.source, args.dataset)
    n_perturbations = int(args.ptb_rate * (adj.sum() // 2))
    num_class = labels.max().item() + 1

    victim_model = GCN(nfeat=features.shape[1], nclass=num_class, nhid=16, dropout=0.5,
                       weight_decay=5e-4, device='cuda').to('cuda')
    victim_model.fit(features, adj, labels, idx_train)
    predicted_labels = victim_model.predict(features, adj).argmax(1)
    # 假设 labels 是 numpy.ndarray，predicted_labels 是 Tensor
    predicted_labels_numpy = predicted_labels.cpu().numpy()  # 将 tensor 转换为 numpy.ndarray
    # 克隆现有的 labels，避免修改原始数据
    pseudo_labels = labels.copy()  # 使用 numpy 的 copy 方法来克隆 labels
    # 使用 numpy 的 isin 函数进行布尔索引替换
    pseudo_labels[~np.isin(np.arange(labels.shape[0]), idx_train)] = predicted_labels_numpy[
        ~np.isin(np.arange(labels.shape[0]), idx_train)]

    pyg_data = load_pyg_data(args.source, args.dataset)
    victim_pyg = load_pyg_model(pyg_data, args.victim, args.source, args.dataset, device, logger, True)

    start_time = time.time()
    model = DICE()
    deltas = []
    for _ in range(combine_n):
        model.attack(adj, pseudo_labels, n_perturbations=n_perturbations)
        # modified_adj = model.modified_adj
        mod_adj = sp.csr_matrix(model.modified_adj)
        selected_idx_test = filter_evasion_pyg(pyg_data, mod_adj, idx_test, select_times, victim_pyg)
        final_filter_adj = get_new_adj(adj, mod_adj, selected_idx_test)
        mod_adj_pyg = retype_adj(final_filter_adj)
        preds_evasion = victim_pyg.predict(pyg_data.x, mod_adj_pyg).argmax(1)
        attack_acc = calculate_accuracy(idx_test, preds_evasion, labels)
        print('filter_acc',attack_acc)
        filter_changed_edges = change_num(adj, final_filter_adj)
        # 过滤扰动数量
        total_filter_edges = n_perturbations - filter_changed_edges
        print('filter_n', total_filter_edges)
        delta = final_filter_adj - adj
        print('delta',delta.nnz/2)
        deltas.append(delta)

    delta_combine = combine_deltas(deltas)
    combine_DICE = adj + delta_combine
    selected_idx_test = filter_evasion_pyg(pyg_data, combine_DICE, idx_test, select_times, victim_pyg)
    combine_filter_adj = get_new_adj(adj, combine_DICE, selected_idx_test)

    changed_edges = change_num(adj, combine_filter_adj)

    mod_adj_pyg = retype_adj(combine_filter_adj)
    victim_pyg = load_pyg_model(pyg_data, args.victim, args.source, args.dataset, device, logger, True)
    preds_evasion = victim_pyg.predict(pyg_data.x, mod_adj_pyg).argmax(1)
    attack_acc = calculate_accuracy(idx_test, preds_evasion, labels)
    print('combine changes', changed_edges,attack_acc)
    end_time = time.time()
    print(f"Time taken to combine deltas: {end_time - start_time:.6f} seconds")

