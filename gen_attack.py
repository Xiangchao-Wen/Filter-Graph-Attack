from utils import *
import argparse
import scipy.sparse as sp
import numpy as np
import torch
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import preprocess
from deeprobust.graph.data import Dataset, PrePtbDataset
from deeprobust.graph.global_attack import Metattack, DICE, Random, MinMax, PGDAttack,MetaApprox  # 请根据具体情况导入模块和函数
from scipy.sparse import csr_matrix
import numpy as np
np.int = int  # 为被弃用的 np.int 创建别名
def get_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        device = f'cuda:{gpu_id}'
    else:
        device = 'cpu'
    return device
def dice_perturbation(predicted_labels, adj, budget, indices, all_indices):
    """
    在无标签数据上实现 DICE 扰动，仅随机添加不同类别的边，针对测试集的部分。

    参数：
    predicted_labels: 预测的标签数组
    adj: 原始邻接矩阵（稀疏矩阵形式）
    budget: 扰动预算（可以增加的边数）
    test_indices: 测试集节点的索引数组

    返回：
    perturb_adj: 扰动后的邻接矩阵
    """
    # 将稀疏矩阵转换为 LIL 格式以便于操作
    perturb_adj = adj.tolil()

    # 获取测试集节点的标签
    test_labels = predicted_labels[indices]

    # 找到不同预测类别的节点对
    different_label_pairs = []
    num_nodes = predicted_labels.shape[0]

    for i in range(len(indices)):
        test_index = indices[i]
        test_label = predicted_labels[test_index]

        # 遍历所有节点
        for other_node in all_indices:
            # 确保这对节点之间没有边，并且它们的标签不同
            if perturb_adj[test_index, other_node] == 0 and perturb_adj[other_node, test_index] == 0:
                if predicted_labels[other_node] != test_label:
                    different_label_pairs.append((test_index, other_node))

    # 随机选择要连接的节点对，限制在预算范围内
    np.random.shuffle(different_label_pairs)
    pairs_to_connect = different_label_pairs[:budget]

    # 添加新的边
    for i, j in pairs_to_connect:
        perturb_adj[i, j] = 1
        perturb_adj[j, i] = 1  # 无向图

    # 返回稀疏矩阵格式
    return perturb_adj.tocsr()

def generate_modified_adj(attack_name, adj, features, labels, idx_train, idx_val, idx_test,
                          name, attack_rate, n_perturbations, num_class, device,pre_trained=False,
                          victim_model=None, pseudo_labels=None):
    """
    根据指定的攻击方法生成被扰动的邻接矩阵 modified_adj。

    参数：
    - attack_name: 要执行的攻击方法列表
    - adj: 原始邻接矩阵
    - features: 节点特征
    - labels: 标签
    - idx_train: 训练集索引
    - idx_val: 验证集索引
    - idx_test: 测试集索引
    - name: 数据集名称
    - attack_rate: 攻击扰动率
    - n_perturbations: 扰动次数
    - num_class: 类别数量
    - pre_trained: 是否使用预训练模型
    - victim_model: 受害模型（用于MinMax和PGDAttack）
    - pseudo_labels: 伪标签（用于DICE_train攻击）

    返回：
    - modified_adj: 被扰动的邻接矩阵（CSR格式）
    """
    modified_adj = None

    # 如果是 MinMax 或 PGDAttack，需要预处理 adj 和 features
    if attack_name in {'MinMax', 'PGDAttack'}:
        adj_a, features_a, labels_a = preprocess(adj, features, labels, preprocess_adj=False)

    if attack_name == 'Metattack':
        if pre_trained:
            perturbed_data = PrePtbDataset(root='/tmp/', name=name, attack_method='meta', ptb_rate=attack_rate)
            modified_adj = perturbed_data.adj
        else:
            surrogate = GCN(nfeat=features.shape[1], nclass=num_class, nhid=16, dropout=0,
                            with_relu=False, with_bias=False, device=device).to(device)
            surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
            model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                              attack_structure=True, attack_features=False, device=device, lambda_=0).to(device)
            model.attack(features, adj, labels, idx_train, idx_test, n_perturbations=n_perturbations, ll_constraint=False)
            # modified_adj = model.modified_adj
            modified_adj = sp.csr_matrix(model.modified_adj.cpu().numpy())

    elif attack_name == 'DICE':
        model = DICE()
        adj_ori = sp.csr_matrix(adj)
        model.attack(adj_ori, labels, n_perturbations=n_perturbations)
        # modified_adj = model.modified_adj
        modified_adj = sp.csr_matrix(model.modified_adj)

    elif attack_name == 'Random':
        model = Random()
        adj_ori = sp.csr_matrix(adj)
        model.attack(adj_ori, labels, n_perturbations=n_perturbations)
        # modified_adj = model.modified_adj
        modified_adj = sp.csr_matrix(model.modified_adj)

    elif attack_name == 'DICE_train':
        adj_ori = sp.csr_matrix(adj)
        idx_all = np.union1d(idx_train, idx_test)
        modified_adj = dice_perturbation(pseudo_labels, adj_ori, n_perturbations, idx_train, idx_all)

    elif attack_name == 'MinMax':
        model = MinMax(model=victim_model, nnodes=adj_a.shape[0], loss_type='CE', device=device).to(device)
        model.attack(features_a, adj_a, labels_a, idx_train, n_perturbations=n_perturbations)
        modified_adj = sp.csr_matrix(model.modified_adj.cpu().numpy())

    elif attack_name == 'PGDAttack':
        model = PGDAttack(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device).to(device)
        model.attack(features_a, adj_a, labels_a, idx_train, n_perturbations=n_perturbations)
        modified_adj = sp.csr_matrix(model.modified_adj.cpu().numpy())

    return modified_adj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pubmed',
                        choices=['cora', 'citeseer', 'pubmed', 'cora_ml', 'polblogs', 'acm','blogcatalog'])
    parser.add_argument('--attack', type=str, default='Metattack',
                        choices=['Metattack', 'MinMax', 'PGDAttack', 'DICE', 'DICE_train', 'random'])
    parser.add_argument('--source', type=str, default='prognn', choices=['planepoied', 'gcn', 'prognn'])
    parser.add_argument('--ptb_rate', type=float, default=0.1)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    device = get_device(args.gpu_id)

    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.source, args.dataset)
    n_perturbations = int(args.ptb_rate * (adj.sum() // 2))
    num_class = labels.max().item() + 1

    victim_model = GCN(nfeat=features.shape[1], nclass=num_class, nhid=16, dropout=0.5,
                       weight_decay=5e-4, device=device).to(device)
    victim_model.fit(features, adj, labels, idx_train)

    predicted_labels = victim_model.predict(features, adj).argmax(1)

    # 假设 labels 是 numpy.ndarray，predicted_labels 是 Tensor
    predicted_labels_numpy = predicted_labels.cpu().numpy()  # 将 tensor 转换为 numpy.ndarray

    # 克隆现有的 labels，避免修改原始数据
    pseudo_labels = labels.copy()  # 使用 numpy 的 copy 方法来克隆 labels

    # 使用 numpy 的 isin 函数进行布尔索引替换
    pseudo_labels[~np.isin(np.arange(labels.shape[0]), idx_train)] = predicted_labels_numpy[
        ~np.isin(np.arange(labels.shape[0]), idx_train)]

    modified_adj = generate_modified_adj(
        attack_name=args.attack,
        adj=adj,
        features=features,
        labels=labels,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        name=args.dataset,
        attack_rate=args.ptb_rate,
        n_perturbations=n_perturbations,
        num_class=num_class,
        victim_model=victim_model,
        pseudo_labels=pseudo_labels,
        pre_trained=True,
        device=device
    )

    if args.source == 'planepoied':
        attack_path = './attack_data/planepoied/adj'

    elif args.source == 'gcn':
        attack_path = './attack_data/deeprobust/gcn/adj'
    else:
        attack_path = './attack_data/deeprobust/prognn/adj'

    filename = args.attack + '-' + args.dataset + '-' + f'{args.ptb_rate}' + '.npz'

    save_path = os.path.join(attack_path, filename)
    # 保存 modified_adj 为 .npz 文件
    sp.save_npz(save_path, modified_adj)
