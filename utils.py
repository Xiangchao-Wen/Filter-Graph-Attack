import os
import torch
from scipy.sparse import csr_matrix
import random
import numpy as np
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import scipy.sparse as sp
from deeprobust.graph.data import Dataset, Dpr2Pyg
import torch.nn as nn
from deeprobust.graph.defense import GCN
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from deeprobust.graph import utils as _utils
from torch_sparse import SparseTensor
import copy
from models import model_map, choice_map
from scipy.sparse import coo_matrix
from tqdm import tqdm  # 引入 tqdm 模块
from ogb.nodeproppred import PygNodePropPredDataset

def get_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        device = f'cuda:{gpu_id}'
    else:
        device = 'cpu'
    return device
def retype_adj(mod_adj):
    adj_coo = mod_adj.tocoo()
    row = torch.tensor(adj_coo.row, dtype=torch.long)
    col = torch.tensor(adj_coo.col, dtype=torch.long)
    value = torch.tensor(adj_coo.data, dtype=torch.float32)
    return SparseTensor(row=row, col=col, value=value, sparse_sizes=mod_adj.shape)


def normalize_feature_numpy(x):
    """
    Normalizes a feature matrix (NumPy format) row-wise to ensure each row sums to 1.

    Parameters:
    x (numpy.ndarray): Input feature matrix of shape (n_samples, n_features).

    Returns:
    torch.FloatTensor: Row-normalized feature matrix as a PyTorch Tensor.
    """

    # 计算每行的和，并防止除零
    row_sums = np.array(x.sum(axis=1)).flatten() + 1e-10  # 将行和转为一维数组

    # 对每行归一化
    inv_row_sums = 1.0 / row_sums  # 求每行和的倒数
    diag_inv = csr_matrix(np.diag(inv_row_sums))  # 构造对角稀疏矩阵
    normalized_x = diag_inv @ x  # 稀疏矩阵乘法实现归一化

    # 返回仍为稀疏矩阵格式
    return normalized_x


def normalize_feature_tensor(x):
    x = _utils.to_scipy(x)
    x = _utils.normalize_feature(x)
    x = torch.FloatTensor(np.array(x.todense()))
    return x
def load_pyg_data(source, name='cora', path='./dataset', seed=15, x_normalize=True,subgraph_ratio= 0.1):
    # assert name in ['cora', 'citeseer', 'pubmed', 'cora_ml','polblogs']
    # x_normalize = False if name == 'polblogs' else True
    # freeze_seed(seed)
    if source=='planetoid':
        if name in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(path, name, transform=T.ToSparseTensor(remove_edge_index=False))
        else:
            dataset = Dataset(root=path, name=name, setting='gcn',seed=15)
            dataset = Dpr2Pyg(dataset, transform=T.ToSparseTensor(remove_edge_index=False))
        data = dataset[0]
        data.num_classes = dataset.num_classes
    elif source=='gcn':
        dataset = Dataset(root=path, name=name, setting='gcn',seed=15)
        dataset = Dpr2Pyg(dataset, transform=T.ToSparseTensor(remove_edge_index=False))
        data = dataset[0]
        data.num_classes = dataset.num_classes
    elif source=='prognn':
        if name in ['cora', 'citeseer', 'pubmed', 'cora_ml', 'polblogs']:
            dataset = Dataset(root=path, name=name, setting='prognn')
        else:
            dataset = Dataset(root=path, name=name, setting='gcn',seed=15)
        dataset = Dpr2Pyg(dataset, transform=T.ToSparseTensor(remove_edge_index=False))
        data = dataset[0]
        data.num_classes = dataset.num_classes
    elif source == 'ogb':
        if name == 'arxiv':
            dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/',transform=T.ToSparseTensor())
            dataset_new = PygNodePropPredDataset(name="ogbn-arxiv", root='./arxiv')
            data = dataset[0]
            data.edge_index = dataset_new[0].edge_index

            edge_index = data.edge_index
            # 添加反向边
            edge_index_sym = torch.cat([edge_index, edge_index.flip(0)], dim=1)

            # 去重，确保不会有重复的边
            edge_index_sym = torch.unique(edge_index_sym, dim=1)

            data.adj_t = SparseTensor.from_edge_index(edge_index_sym)

            # 更新数据对象的 edge_index
            data.edge_index = edge_index_sym
            # print('arxiv edge index',data.edge_index)
            # data.edge_index = data.edge_index
            data.num_classes = dataset.num_classes
            data.y = data.y.squeeze(1)
            split_idx = dataset.get_idx_split()
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.train_mask[split_idx['train']] = True
            data.val_mask[split_idx['valid']] = True
            data.test_mask[split_idx['test']] = True
        else:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)  # 为了确保numpy中的随机操作也固定

            dataset = PygNodePropPredDataset(name="ogbn-products", root='./arxiv', transform=T.ToSparseTensor())
            data = dataset[0]
            dataset_new = PygNodePropPredDataset(name="ogbn-products", root='./arxiv')
            split_idx = dataset.get_idx_split()
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.train_mask[split_idx['train']] = True
            data.val_mask[split_idx['valid']] = True
            data.test_mask[split_idx['test']] = True

            data.edge_index = dataset_new[0].edge_index

            # 提取子图节点
            num_nodes = data.num_nodes
            subgraph_nodes = int(num_nodes * subgraph_ratio)
            sampled_nodes = torch.tensor(random.sample(range(num_nodes), subgraph_nodes), dtype=torch.long)

            # 更新子图特征
            data.x = data.x[sampled_nodes]
            data.y = data.y[sampled_nodes]

            # 提取子图的边
            edge_index = data.edge_index
            mask = torch.isin(edge_index[0], sampled_nodes) & torch.isin(edge_index[1], sampled_nodes)
            edge_index_sub = edge_index[:, mask]

            # 重映射边的索引
            mapping = {n.item(): i for i, n in enumerate(sampled_nodes)}
            edge_index_sub[0] = torch.tensor([mapping[n.item()] for n in edge_index_sub[0]], dtype=torch.long)
            edge_index_sub[1] = torch.tensor([mapping[n.item()] for n in edge_index_sub[1]], dtype=torch.long)

            # 更新子图的 edge_index 和 adj_t
            data.edge_index = edge_index_sub
            data.adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(subgraph_nodes, subgraph_nodes))

            data.num_nodes = subgraph_nodes

            # 只保留与采样节点相关的 train/val/test mask
            data.train_mask = data.train_mask[sampled_nodes]
            data.val_mask = data.val_mask[sampled_nodes]
            data.test_mask = data.test_mask[sampled_nodes]

            data.num_classes = dataset.num_classes
            data.y = data.y.squeeze(1)

    if name == 'karate_club':
        data.test_mask = ~(data.train_mask)
        data.val_mask = torch.zeros_like(data.test_mask, dtype=torch.bool)
    if x_normalize:
        data.x = normalize_feature_tensor(data.x)
    return data

def get_logger(filename, level=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[level])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
def calculate_accuracy(idx_test, preds, labels):
    """
    计算并返回原始准确率和攻击后准确率。

    参数:
    idx_test (list): 测试集索引
    preds (list): 攻击前的预测结果
    labels (list): 实际标签
    返回:
    ori_acc (float): 攻击前准确率
    """
    correct = sum([1 for idx in idx_test if preds[idx] == labels[idx]])
    ori_acc = correct / (len(idx_test) + 0.00000001)
    return ori_acc

def change_num(adj,new_adj):
    # 确保输入的 adj 和 perturbed_adj 是稀疏矩阵
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    if not sp.issparse(new_adj):
        new_adj = sp.csr_matrix(new_adj)

    # 计算两个邻接矩阵的差异
    diff_matrix = adj - new_adj

    # 找到差异矩阵中非零的元素（这些是改变了的边）
    changed_edges = np.abs(diff_matrix).nnz // 2  # 因为是无向图，边计算了两次，需要除以2
    return changed_edges

def load_data(source,dataset):
    if dataset in ['cora_ml', 'polblogs'] and source == 'planepoied':
        source = 'prognn'
    if source == 'planepoied':
        path = './dataset'
        dataset = Planetoid(path, dataset, transform=T.ToSparseTensor(remove_edge_index=False))
        data = dataset[0]
        data.num_classes = dataset.num_classes
        pyg_data = data
        edge_index = pyg_data.edge_index
        num_nodes = pyg_data.num_nodes
        adj = sp.csr_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                            shape=(num_nodes, num_nodes))
        features = pyg_data.x.numpy()
        labels = pyg_data.y.numpy()
        idx_train = torch.nonzero(pyg_data.train_mask).squeeze()
        idx_test = torch.nonzero(pyg_data.test_mask).squeeze()
        idx_val = torch.nonzero(pyg_data.val_mask).squeeze()
    elif source == 'gcn':
        data = Dataset(root='./dataset/', name=dataset, setting='gcn', seed=15)
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        adj, features, labels = data.adj, data.features, data.labels
    elif source == 'prognn':
        data = Dataset(root='./dataset/', name=dataset, setting='prognn')
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        adj, features, labels = data.adj, data.features, data.labels
    elif source == 'ogb':
        if dataset=='arxiv':
            dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/')
        elif dataset=='products':
            dataset = PygNodePropPredDataset(name='ogbn-products', root='./arxiv/')
        data = dataset[0]
        features = data.x.numpy()
        labels = data.y.numpy().squeeze()
        # print(labels)

        adj = coo_matrix((np.ones(data.edge_index.shape[1]),
                          (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
                         shape=(data.num_nodes, data.num_nodes))
        # adj = (adj + adj.T)
        # adj.data = np.minimum(adj.data, 1)
        # 确保矩阵对称性
        adj = (adj + adj.T)
        # 限制最大值为 1
        adj.data = np.clip(adj.data, 0, 1)
        adj = adj.tocsr()

        # # 获取边的起点和终点
        # row, col = data.edge_index[0].numpy(), data.edge_index[1].numpy()
        #
        # # 构造对称的边集
        # row_symmetric = np.concatenate([row, col])  # 添加反向边的起点
        # col_symmetric = np.concatenate([col, row])  # 添加反向边的终点
        # values = np.ones(len(row_symmetric))  # 所有边的值设置为1
        #
        # # 构造稀疏矩阵，确保对称性
        # adj = coo_matrix((values, (row_symmetric, col_symmetric)), shape=(data.num_nodes, data.num_nodes))

        # 获取官方划分的训练、验证、测试集索引
        split_idx = dataset.get_idx_split()
        idx_train = split_idx['train']
        idx_val = split_idx['valid']
        idx_test = split_idx['test']
        if dataset=='products':
            torch.manual_seed(15)
            np.random.seed(15)  # 确保numpy中的随机操作也固定

            # 计算需要采样的节点数
            num_nodes = data.num_nodes
            subgraph_nodes = int(num_nodes * 0.1)

            # 使用torch的随机采样方法，保证固定性
            sampled_nodes = torch.randperm(num_nodes)[:subgraph_nodes]

            # 更新子图特征和标签
            data.x = data.x[sampled_nodes]
            data.y = data.y[sampled_nodes]

            # 更新邻接矩阵
            mask = torch.isin(data.edge_index[0], sampled_nodes) & torch.isin(data.edge_index[1], sampled_nodes)
            edge_index_sub = data.edge_index[:, mask]

            # 重映射边的索引
            mapping = {n.item(): i for i, n in enumerate(sampled_nodes)}
            edge_index_sub[0] = torch.tensor([mapping[n.item()] for n in edge_index_sub[0]], dtype=torch.long)
            edge_index_sub[1] = torch.tensor([mapping[n.item()] for n in edge_index_sub[1]], dtype=torch.long)

            # 更新子图的 edge_index 和 adj_t
            data.edge_index = edge_index_sub
            data.adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(subgraph_nodes, subgraph_nodes))

            data.num_nodes = subgraph_nodes

            # 只保留与采样节点相关的 train/val/test mask
            data.train_mask = data.train_mask[sampled_nodes]
            data.val_mask = data.val_mask[sampled_nodes]
            data.test_mask = data.test_mask[sampled_nodes]

    # print('feature',type(features),features)
    # features = normalize_feature_numpy(features)
    return adj, features, labels, idx_train, idx_val, idx_test

def load_attack_adj(source,attack_method,dataset,ptb_rate):
    assert source in ['planepoied', 'gcn', 'prognn', 'ogb']
    if dataset in ['cora_ml','polblogs']:
        if source == 'planepoied':
            source = 'gcn'
    if source == 'planepoied':
        attack_path = './attack_data/planepoied/adj'

    elif source == 'gcn':
        attack_path = './attack_data/deeprobust/gcn/adj'
    elif source=='prognn':
        attack_path = './attack_data/deeprobust/prognn/adj'
    elif source=='ogb':
        attack_path = '../PGA-main/attack/perturbed_adjs/ogb'


    # if source=='prognn' and attack_method=='pga' and dataset=='cora':
    #     attack_path = '../PGA-main/attack/perturbed_adjs'

    if attack_method in ['greedy','prbcd','pga','pgdattack-CW']:
        attack_path = '../PGA-main/attack/perturbed_adjs/gcn'
        if source == 'ogb':
            attack_path = '../PGA-main/attack/perturbed_adjs/ogb'
        filename = attack_method + '-' + dataset + '-' + f'{ptb_rate}' + '.pth'
        filename = os.path.join(attack_path, filename)
        data = torch.load(filename)
        modified_adj_list = data['modified_adj_list']
        # attack_config = data['attack_config']
        mod_adj = modified_adj_list[0]
        # 获取 COO 格式的行、列索引和非零值
        row, col, values = mod_adj.coo()
        if values is None:
            values = torch.ones(row.size(0))  # 生成与非零元素个数一致的权重
        # 获取 shape 信息
        num_rows, num_cols = mod_adj.sizes()

        # 转换为 CSR 格式
        csr_mod_adj = csr_matrix((values.numpy(), (row.cpu().numpy(), col.cpu().numpy())),
                                 shape=(num_rows, num_cols))

        # 将 csr_mod_adj 转换为 numpy 数组
        csr_mod_adj_np = csr_mod_adj
        # csr_mod_adj_np = csr_mod_adj.toarray()  # 如果 csr_mod_adj 是 CSR 格式
    elif attack_method in ['Metattack', 'MinMax', 'PGDAttack', 'DICE', 'DICE_train',]:
        try:
            filename = attack_method + '-' + dataset + '-' + f'{ptb_rate}' + '.npz'
            filename = os.path.join(attack_path, filename)
            csr_mod_adj_np = sp.load_npz(filename)
        except:
            filename = attack_method + '-' + dataset + '-' + f'{ptb_rate}' + '.npy'
            filename = os.path.join(attack_path, filename)
            csr_mod_adj_np = np.load(filename)
    return csr_mod_adj_np


def save_adj_to_npz(source, attack_method, dataset, ptb_rate, adj, adj_source, attack_rate):
    # 验证source输入是否合法
    assert source in ['planepoied', 'gcn', 'prognn','ogb'], "Invalid source!"

    # 验证adj_source输入是否合法
    # assert adj_source in ['break_adj', 'final_adj'], "Invalid adj_source!"

    # 设置文件路径
    if source == 'planepoied':
        attack_path = './filter_data/planepoied/adj'
    elif source == 'gcn':
        attack_path = './filter_data/deeprobust/gcn/adj'
    elif source == 'prognn':
        attack_path = './filter_data/deeprobust/prognn/adj'
    else:
        attack_path = './filter_data/ogb/adj'

    # 构建文件名，包含adj_source信息
    filename = f"{attack_method}-{dataset}-{ptb_rate}-{adj_source}-{attack_rate}.npz"
    filepath = os.path.join(attack_path, filename)

    # 确保目标路径存在
    os.makedirs(attack_path, exist_ok=True)

    # 将 adj 转换为 CSR 格式并保存为 .npz 文件
    if isinstance(adj, csr_matrix):
        sp.save_npz(filepath, adj)
    else:
        # 如果 adj 不是 CSR 格式，先转为 CSR 格式
        csr_adj = csr_matrix(adj)
        # np.savez(filepath, adj=csr_adj)
        sp.save_npz(filepath, csr_adj)

    print(f"Adjacency matrix saved to {filepath}")


def load_adj_from_npz(source, attack_method, dataset, ptb_rate, adj_source):
    # 验证source输入是否合法
    assert source in ['planepoied', 'gcn', 'prognn','ogb'], "Invalid source!"

    # 验证adj_source输入是否合法
    assert adj_source in ['break_adj', 'final_adj'], "Invalid adj_source!"

    # 处理dataset和source的特殊关系
    if dataset in ['cora_ml', 'polblogs']:
        if source == 'planepoied':
            source = 'prognn'

    # 设置文件路径
    if source == 'planepoied':
        attack_path = './filter_data/planepoied/adj'
    elif source == 'gcn':
        attack_path = './filter_data/deeprobust/gcn/adj'
    elif source=='prognn':
        attack_path = './filter_data/deeprobust/prognn/adj'
    else:
        attack_path = './filter_data/ogb/adj'

    # 构建文件名，包含adj_source信息
    filename = f"{attack_method}-{dataset}-{ptb_rate}-{adj_source}.npz"
    filepath = os.path.join(attack_path, filename)

    # 检查文件是否存在
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist!")

    # 读取 .npz 文件
    adj = sp.load_npz(filepath)
    # 如果是CSR格式的矩阵，直接返回；否则将其转换为CSR格式
    if isinstance(adj, csr_matrix):
        return adj
    else:
        # 如果矩阵不是CSR格式，转换为CSR格式
        return csr_matrix(adj)

def load_pyg_model(pyg_data,model_name,source,dataset,device,logger,pretrained):
    # model_name = args.victim
    save_path = f"../PGA-main/victims/models/{source}/"
    save_path += f"{model_name + '-' + dataset}.pth"
    savings = torch.load(save_path)
    config = savings['config']
    states = savings['state_dicts']
    performance = savings['performance']
    model_func = model_map[model_name]
    model = model_func(config=config, pyg_data=pyg_data, device=device, logger=logger)
    model = model.to(device)
    model.load_state_dict(states[0])
    if pretrained == False:
        model.fit(pyg_data)
    return model
def filter_poison_pyg(pyg_data, mod_adj, repeat_n, victim, filter_method):
    labels = pyg_data.y.numpy()
    idx_train = torch.nonzero(pyg_data.train_mask).squeeze()
    idx_test = torch.nonzero(pyg_data.test_mask).squeeze()

    incorrect_after_perturbation = set()
    poison_pyg_data = copy.deepcopy(pyg_data)
    poison_pyg_data.adj_t = mod_adj
    for _ in range(repeat_n):
        print(_,repeat_n)
        # clean graph
        # victim.fit(pyg_data, verbose=True)
        preds_clean = victim.predict(pyg_data.x, pyg_data.adj_t).argmax(1)
        preds_evasion = victim.predict(pyg_data.x, mod_adj).argmax(1)
        # baseline
        victim.fit(poison_pyg_data, verbose=False)
        # 在测试集上进行预测，并获取预测的概率分布
        preds_poison = victim.predict(pyg_data.x, mod_adj).argmax(1)
        preds_vasion = victim.predict(pyg_data.x, pyg_data.adj_t).argmax(1)
        # 保留在原始矩阵上预测正确的节点，并筛选出在扰动矩阵上预测错误的节点
        if filter_method == 'test':
            for idx in idx_test:  # 只考虑 idx_test 中的节点
                if preds_vasion[idx] != preds_clean[idx]:
                    incorrect_after_perturbation.add(idx)
        # 如果考虑，则保留过多的扰动，对过滤不利
        if filter_method == 'train':
            for idx in idx_train:  # 考虑 idx_train 中的节点
                if preds_evasion[idx] != labels[idx]:
                    incorrect_after_perturbation.add(idx)
    return list(incorrect_after_perturbation)  # 将 set 转换为列表并返回
# 筛选有用扰动的节点列表
def filter_poison(adj, mod_adj, features, idx_train, idx_val, idx_test, labels, repeat_n, device):
    incorrect_after_perturbation = set()
    for _ in range(repeat_n):
        print(_,repeat_n)
        # clean graph
        gcn = GCN(nfeat=features.shape[1],
            nhid=16,
            nclass=labels.max().item() + 1,
            device=device)
        gcn = gcn.to(device)
        gcn.fit(features, adj, labels, idx_train, idx_val) # train on clean graph with earlystopping
        preds_before = gcn.predict(features, adj).argmax(1)
        # baseline
        gcn_poison = GCN(nfeat=features.shape[1],
                         nhid=16,
                         nclass=labels.max().item() + 1,
                         device=device)
        gcn_poison = gcn_poison.to(device)
        gcn_poison.fit(features, mod_adj, labels, idx_train, idx_val)  # train on clean graph with earlystopping
        # 在测试集上进行预测，并获取预测的概率分布
        preds_after = gcn_poison.predict(features, mod_adj).argmax(1)
        # 保留在原始矩阵上预测正确的节点，并筛选出在扰动矩阵上预测错误的节点
        for idx in idx_test:  # 只考虑 idx_test 中的节点
            if preds_before[idx] != preds_after[idx]:
                incorrect_after_perturbation.add(idx)
        # 如果考虑，则保留过多的扰动，对过滤不利
        # for idx in idx_train:  # 考虑 idx_train 中的节点
        #     if preds_before[idx] != preds_after[idx]:
        #         incorrect_after_perturbation.add(idx)
    return list(incorrect_after_perturbation)  # 将 set 转换为列表并返回
def filter_evasion_pyg(pyg_data,mod_adj, idx_test, repeat_n, victim):
    device = 'cuda'
    incorrect_after_perturbation = set()
    for i in range(repeat_n):
        print(i)
        victim.fit(pyg_data)
        preds_before = victim.predict(pyg_data.x, pyg_data.adj_t).argmax(1)
        adj_coo = mod_adj.tocoo()
        edge_index = torch.tensor([adj_coo.row, adj_coo.col], dtype=torch.long)
        preds_after = victim.predict(pyg_data.x, edge_index.to(device)).argmax(1)
        # 保留在原始矩阵上预测正确的节点，并筛选出在扰动矩阵上预测错误的节点
        for idx in idx_test:  # 只考虑 idx_test 中的节点
            if preds_before[idx] != preds_after[idx]:
                incorrect_after_perturbation.add(idx)

    return list(incorrect_after_perturbation)  # 将 set 转换为列表并返回
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
        adj[:, batch_idx] = adj_new[:, batch_idx]

    # 确保矩阵对称性
    adj = (adj + adj.T)
    adj.data = np.clip(adj.data, 0, 1)

    # 返回转回 CSR 格式的稀疏矩阵
    return adj.tocsr()  # 转换回 CSR 格式进行后续处理


def filter_edge_combined_pyg_batch(gcn, gcn_poison, adj, mod_adj, pyg_data, filter_num, epsilon, gpu_id, confidence_score, batch_size=10, lamda=0.05):
    labels = pyg_data.y.numpy()
    idx_train = torch.nonzero(pyg_data.train_mask).squeeze()
    idx_test = torch.nonzero(pyg_data.test_mask).squeeze()
    features = pyg_data.x.numpy()

    search_count = -1
    # Define the cross-entropy loss function
    criterion = nn.CrossEntropyLoss(reduction='none')
    labels_tensor = torch.tensor(labels, dtype=torch.long).cuda(gpu_id)

    # Convert adj matrices to LIL format for efficient manipulation (return CSR at the end)
    adj = adj.tolil()
    mod_adj = mod_adj.tolil()
    perturb_adj = mod_adj.copy()

    # 获取 mod_adj 的形状
    shape = mod_adj.shape
    # 创建一个与 mod_adj 大小相同但所有元素为零的稀疏矩阵
    break_adj = csr_matrix(shape)

    # Calculate the difference between adj and mod_adj
    adj_diff = mod_adj - adj  # Positive: deleted edge, Negative: added edge

    # Repeat for the specified number of times to remove perturbations
    for file_n in range(0,filter_num,batch_size):
        found_valid_edges = False  # 标志变量，用于指示是否找到符合条件的边
        while not found_valid_edges:
            # Get predicted logits and compute losses for clean and poisoned models
            perturb_adj_pyg = retype_adj(perturb_adj)
            logits_clean = gcn.predict(pyg_data.x, perturb_adj_pyg)
            loss_clean = criterion(logits_clean, labels_tensor).cpu().detach().numpy()
            logits_poison = gcn_poison.predict(pyg_data.x, perturb_adj_pyg)
            loss_poison = criterion(logits_poison, labels_tensor).cpu().detach().numpy()

            # Calculate scores for each perturbation edge
            scores = []
            rows, cols = adj_diff.nonzero()  # Get rows, columns of perturbed edges
            for i, j in zip(rows, cols):
                if i < j:
                    loss_i = loss_clean[i] if i in idx_train else confidence_score[i] * loss_poison[i]
                    loss_j = loss_clean[j] if j in idx_train else confidence_score[j] * loss_poison[j]
                    score = loss_i + loss_j
                    scores.append((score, i, j))

            # Sort scores to start with the smallest perturbation score
            if scores:
                scores.sort(key=lambda x: x[0])

                # Try multiple batches of edges
                for batch_start in range(0, len(scores) - batch_size + 1):
                    # Select the batch of edges to modify
                    selected_edges = scores[batch_start: batch_start + batch_size]

                    # Create a copy of perturb_adj to apply modifications
                    perturb_adj_temp = perturb_adj.copy()

                    # Apply changes to the selected edges
                    for _, node_i, node_j in selected_edges:
                        perturb_adj_temp[node_i, node_j] = 1 - perturb_adj_temp[node_i, node_j]
                        perturb_adj_temp[node_j, node_i] = 1 - perturb_adj_temp[node_j, node_i]

                    # Recalculate loss_poison for selected nodes (N-hop neighbors or all nodes)
                    perturb_adj_pyg = retype_adj(perturb_adj_temp)
                    logits_poison_temp = gcn_poison.predict(pyg_data.x, perturb_adj_pyg)
                    loss_poison_temp = criterion(logits_poison_temp, labels_tensor).cpu().detach().numpy()

                    # Compute average loss over the selected nodes
                    # avg_loss_poison_temp = np.mean([loss_poison_temp[n] for n in idx_train])
                    # avg_loss_poison_original = np.mean([loss_poison[n] for n in idx_train])

                    avg_loss_poison_temp = np.mean([
                        loss_poison_temp[n] if n in idx_train else confidence_score[n] * loss_poison_temp[n]
                        for n in range(len(loss_poison_temp))
                    ])
                    avg_loss_poison_original = np.mean([
                        loss_poison[n] if n in idx_train else confidence_score[n] * loss_poison[n]
                        for n in range(len(loss_poison))
                    ])

                    # avg_loss_poison_temp = np.mean(np.where(pyg_data.train_mask, loss_poison_temp, lamda * loss_poison_temp))
                    # avg_loss_poison_original = np.mean(np.where(pyg_data.train_mask, loss_poison, lamda * loss_poison))

                    # Check if the average loss for the selected nodes increases
                    if avg_loss_poison_temp > (1 - epsilon) * avg_loss_poison_original:
                        # Permanently apply the edge modifications if condition is met
                        perturb_adj = perturb_adj_temp.copy()
                        # Update adj_diff for the modified edges
                        for _, node_i, node_j in selected_edges:
                            adj_diff[node_i, node_j] = 0
                            adj_diff[node_j, node_i] = 0
                        print('过滤边数',file_n,'搜索次数和总数',batch_start,len(scores))
                        found_valid_edges = True  # 找到符合条件的边，设置标志为 True
                        break  # Stop after the first valid modification batch
                # If no valid edges in this batch, continue to next batch
                if found_valid_edges:
                    break

                # If no valid edges are found after a certain number of attempts, adjust epsilon
                if not found_valid_edges:
                    search_count = file_n
                    break_adj = perturb_adj.copy()
                    epsilon += 0.001
                    print("No valid edge found, epsilon has changed to ", epsilon)
                    break

    # Return the modified adjacency matrix in CSR format
    return perturb_adj.tocsr(), epsilon, search_count, break_adj


def filter_edge_combined_pyg(gcn, gcn_poison, adj, mod_adj, pyg_data, filter_num, epsilon, gpu_id):
    labels = pyg_data.y.numpy()
    idx_train = torch.nonzero(pyg_data.train_mask).squeeze()
    idx_test = torch.nonzero(pyg_data.test_mask).squeeze()
    features = pyg_data.x.numpy()

    search_count = -1
    # Define the cross-entropy loss function
    criterion = nn.CrossEntropyLoss(reduction='none')
    labels_tensor = torch.tensor(labels, dtype=torch.long).cuda(gpu_id)

    # Convert adj matrices to LIL format for efficient manipulation (return CSR at the end)
    adj = adj.tolil()
    mod_adj = mod_adj.tolil()
    perturb_adj = mod_adj.copy()

    # 获取 mod_adj 的形状
    shape = mod_adj.shape
    # 创建一个与 mod_adj 大小相同但所有元素为零的稀疏矩阵
    break_adj = csr_matrix(shape)

    # Calculate the difference between adj and mod_adj
    adj_diff = mod_adj - adj  # Positive: deleted edge, Negative: added edge

    # Repeat for the specified number of times to remove perturbations
    for file_n in range(filter_num):
        found_valid_edge = False  # 标志变量，用于指示是否找到符合条件的边
        while not found_valid_edge:
            # Get predicted logits and compute losses for clean and poisoned models
            perturb_adj_pyg = retype_adj(perturb_adj)
            logits_clean = gcn.predict(pyg_data.x, perturb_adj_pyg)
            loss_clean = criterion(logits_clean, labels_tensor).cpu().detach().numpy()
            logits_poison = gcn_poison.predict(pyg_data.x, perturb_adj_pyg)
            loss_poison = criterion(logits_poison, labels_tensor).cpu().detach().numpy()
            # Calculate scores for each perturbation edge
            scores = []
            rows, cols = adj_diff.nonzero()  # Get rows, columns of perturbed edges
            for i, j in zip(rows, cols):
                if i != j:
                    loss_i = loss_clean[i] if i in idx_train else loss_poison[i]
                    loss_j = loss_clean[j] if j in idx_train else loss_poison[j]
                    score = loss_i + loss_j
                    scores.append((score, i, j))

            # Sort scores to start with the smallest perturbation score
            if scores:
                scores.sort(key=lambda x: x[0])
                found_valid_edge = False  # 标志变量，用于指示是否找到符合条件的边
                # Iterate through sorted scores to find the first edge that meets the condition
                for _, node_i, node_j in scores:
                    perturb_adj[node_i, node_j] = 1 - perturb_adj[node_i, node_j]
                    perturb_adj[node_j, node_i] = 1 - perturb_adj[node_j, node_i]

                    # Determine the nodes for calculating the average loss based on use_n_hop
                    neighbors = range(adj.shape[0])

                    # Recalculate loss_poison for selected nodes (N-hop neighbors or all nodes)
                    perturb_adj_pyg = retype_adj(perturb_adj)
                    logits_poison_temp = gcn_poison.predict(pyg_data.x, perturb_adj_pyg)
                    loss_poison_temp = criterion(logits_poison_temp, labels_tensor).cpu().detach().numpy()

                    # Compute average loss over the selected nodes
                    avg_loss_poison_temp = np.mean([loss_poison_temp[n] for n in neighbors])
                    avg_loss_poison_original = np.mean([loss_poison[n] for n in neighbors])

                    # Check if the average loss for the selected nodes increases

                    if avg_loss_poison_temp > (1 - epsilon) * avg_loss_poison_original:
                        # Permanently apply the edge modification
                        adj_diff[node_i, node_j] = 0
                        adj_diff[node_j, node_i] = 0
                        print('search_count',_)
                        found_valid_edge = True  # 找到符合条件的边，设置标志为 True
                        break  # Stop after the first valid modification
                    else:
                        perturb_adj[node_i, node_j] = 1 - perturb_adj[node_i, node_j]
                        perturb_adj[node_j, node_i] = 1 - perturb_adj[node_j, node_i]
                # 如果所有尝试的边都不符合条件，退出主循环
                if not found_valid_edge:
                    search_count = file_n
                    break_adj = perturb_adj.copy()
                    epsilon += 0.0001
                    print("No valid edge found, epsilon has changed to ",epsilon)
                # break
    # Return the modified adjacency matrix in CSR format
    return perturb_adj.tocsr(), epsilon, search_count, break_adj

def filter_edge_combined(gcn, gcn_poison, adj, mod_adj, idx_train, idx_test, features, labels, filter_num, epsilon, gpu_id):
    search_count = -1
    # Define the cross-entropy loss function
    criterion = nn.CrossEntropyLoss(reduction='none')
    labels_tensor = torch.tensor(labels, dtype=torch.long).cuda(gpu_id)

    # Convert adj matrices to LIL format for efficient manipulation (return CSR at the end)
    adj = adj.tolil()
    mod_adj = mod_adj.tolil()
    perturb_adj = mod_adj.copy()

    # 获取 mod_adj 的形状
    shape = mod_adj.shape
    # 创建一个与 mod_adj 大小相同但所有元素为零的稀疏矩阵
    break_adj = csr_matrix(shape)

    # Calculate the difference between adj and mod_adj
    adj_diff = mod_adj - adj  # Positive: deleted edge, Negative: added edge

    # Repeat for the specified number of times to remove perturbations
    for file_n in range(filter_num):
        found_valid_edge = False  # 标志变量，用于指示是否找到符合条件的边
        while not found_valid_edge:
            # Get predicted logits and compute losses for clean and poisoned models
            logits_clean = gcn.predict(features, perturb_adj)
            loss_clean = criterion(logits_clean, labels_tensor).cpu().detach().numpy()
            logits_poison = gcn_poison.predict(features, perturb_adj)
            loss_poison = criterion(logits_poison, labels_tensor).cpu().detach().numpy()
            # Calculate scores for each perturbation edge
            scores = []
            rows, cols = adj_diff.nonzero()  # Get rows, columns of perturbed edges
            for i, j in zip(rows, cols):
                if i != j:
                    loss_i = loss_clean[i] if i in idx_train else loss_poison[i]
                    loss_j = loss_clean[j] if j in idx_train else loss_poison[j]
                    score = loss_i + loss_j
                    scores.append((score, i, j))

            # Sort scores to start with the smallest perturbation score
            if scores:
                scores.sort(key=lambda x: x[0])
                found_valid_edge = False  # 标志变量，用于指示是否找到符合条件的边
                # Iterate through sorted scores to find the first edge that meets the condition
                for _, node_i, node_j in scores:
                    perturb_adj[node_i, node_j] = 1 - perturb_adj[node_i, node_j]
                    perturb_adj[node_j, node_i] = 1 - perturb_adj[node_j, node_i]

                    # Determine the nodes for calculating the average loss based on use_n_hop
                    neighbors = range(adj.shape[0])

                    # Recalculate loss_poison for selected nodes (N-hop neighbors or all nodes)
                    logits_poison_temp = gcn_poison.predict(features, perturb_adj)
                    loss_poison_temp = criterion(logits_poison_temp, labels_tensor).cpu().detach().numpy()

                    # Compute average loss over the selected nodes
                    avg_loss_poison_temp = np.mean([loss_poison_temp[n] for n in neighbors])
                    avg_loss_poison_original = np.mean([loss_poison[n] for n in neighbors])

                    # Check if the average loss for the selected nodes increases

                    if avg_loss_poison_temp > (1 - epsilon) * avg_loss_poison_original:
                        # Permanently apply the edge modification
                        adj_diff[node_i, node_j] = 0
                        adj_diff[node_j, node_i] = 0
                        print('search_count',_)
                        found_valid_edge = True  # 找到符合条件的边，设置标志为 True
                        break  # Stop after the first valid modification
                    else:
                        perturb_adj[node_i, node_j] = 1 - perturb_adj[node_i, node_j]
                        perturb_adj[node_j, node_i] = 1 - perturb_adj[node_j, node_i]
                # 如果所有尝试的边都不符合条件，退出主循环
                if not found_valid_edge:
                    search_count = file_n
                    break_adj = perturb_adj.copy()
                    epsilon += 0.0001
                    print("No valid edge found, epsilon has changed to ",epsilon)
                # break
    # Return the modified adjacency matrix in CSR format
    return perturb_adj.tocsr(), epsilon, search_count, break_adj

def toggle_attack_edges(adj, mod_adj, filter_num):
    """
    Randomly toggles `filter_num` attack edges in `mod_adj`. If the edge was added, it will be removed.
    If the edge was removed, it will be restored.

    Parameters:
    adj (csr_matrix): Original adjacency matrix.
    mod_adj (csr_matrix): Perturbed adjacency matrix with attack edges.
    filter_num (int): Number of attack edges to randomly toggle.

    Returns:
    filter_adj (csr_matrix): Modified adjacency matrix after toggling attack edges.
    """
    # Convert adjacency matrices to LIL format for efficient manipulation
    adj = adj.tolil()
    mod_adj = mod_adj.tolil()
    filter_adj = mod_adj.copy()

    # Calculate the difference to identify attack edges (1 in mod_adj but 0 in adj for added edges, and vice versa)
    added_edges = np.array((mod_adj - adj).nonzero()).T  # Edges added in mod_adj
    removed_edges = np.array((adj - mod_adj).nonzero()).T  # Edges removed in mod_adj

    # Combine all attack edges (added + removed) for random selection
    attack_edges = np.vstack((added_edges, removed_edges))

    # Use a set to keep track of unique edges (only store one direction)
    unique_edges = set()
    filtered_attack_edges = []

    for i, j in attack_edges:
        if (j, i) not in unique_edges:  # Check if the reverse edge is already in the set
            unique_edges.add((i, j))
            unique_edges.add((j, i))
            filtered_attack_edges.append((i, j))

    # Convert filtered_attack_edges to a numpy array
    attack_edges = np.array(filtered_attack_edges)

    # If filter_num exceeds available attack edges, adjust it
    if filter_num > len(attack_edges):
        filter_num = len(attack_edges)

    # Randomly select `filter_num` attack edges
    selected_edges = attack_edges[np.random.choice(len(attack_edges), filter_num, replace=False)]
    # print('filter_n:',len(selected_edges))
    # Toggle each selected edge
    for i, j in selected_edges:
        if filter_adj[i, j] == 1:  # If edge is present in mod_adj but not in adj, remove it
            filter_adj[i, j] = 0
            filter_adj[j, i] = 0  # Symmetric removal for undirected graph
        else:  # If edge is missing in mod_adj but present in adj, restore it
            filter_adj[i, j] = 1
            filter_adj[j, i] = 1  # Symmetric restoration for undirected graph

    # Return modified adjacency matrix in CSR format
    return filter_adj.tocsr()
def save_results_poison(source, name, method_name, attack_rate, ori_changed_edges,
                          filter_changes, edge_acc_result, filter_acc_result, random_acc_result, break_count,
                          repeat_n=10):
    # Define folder paths and filenames
    output_folder = f'./poison_result/{source}'
    os.makedirs(output_folder, exist_ok=True)
    file_name = f"{name}_{method_name}_{attack_rate}_{ori_changed_edges}_new.csv"
    file_path = os.path.join(output_folder, file_name)

    # Write data to CSV
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Modify header to include break_count
        header = ['Change', 'Break_Count'] + \
                 [f'Edge_filter_{i + 1}' for i in range(repeat_n)] + \
                 [f'Target_filter_{i + 1}' for i in range(repeat_n)] + \
                 [f'Random_filter_{i + 1}' for i in range(repeat_n)]
        writer.writerow(header)

        # Write rows with break_count value
        for i in range(len(filter_changes)):
            row = [filter_changes[i], break_count] + edge_acc_result[i] + filter_acc_result[i] + random_acc_result[i]
            writer.writerow(row)

def save_results_evasion(source, name, method_name, attack_rate, ori_changed_edges,
                          ori_acc_list, attack_acc_list, filter_acc_list, random_acc_list,
                         filter_changed_edges_list):
    # Define folder paths and filenames
    output_folder = f'./evasion_result/{source}'
    os.makedirs(output_folder, exist_ok=True)
    file_name = f"{name}_{method_name}_{attack_rate}_{ori_changed_edges}.csv"
    file_path = os.path.join(output_folder, file_name)

    # 创建一个字典，将所有结果存储在一起
    data = {
        'filter_attack_num': filter_changed_edges_list,
        'ori_acc': ori_acc_list,
        'attack_acc': attack_acc_list,
        'filter_acc': filter_acc_list,
        'random_acc': random_acc_list
    }

    # 将字典转换为DataFrame
    df = pd.DataFrame(data)

    # 保存DataFrame到CSV文件
    df.to_csv(file_path, index=False)

    # # Write data to CSV
    # with open(file_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #
    #     # Modify header to include break_count
    #     header = ['Change'] + \
    #              [f'Efficacy_filter_{i + 1}' for i in range(repeat_n)] + \
    #              [f'Random_filter_{i + 1}' for i in range(repeat_n)]
    #     writer.writerow(header)
    #
    #     # Write rows with break_count value
    #     for i in range(len(filter_changes)):
    #         row = filter_changes[i] + filter_acc_result[i] + random_acc_result[i]
    #         writer.writerow(row)

def save_results_and_plot(source, name, method_name, attack_rate, ori_changed_edges,
                          filter_changes, edge_acc_result, filter_acc_result, random_acc_result, break_count,
                          repeat_n=10):
    """
    Saves the results to a CSV file and generates a plot comparing filter and random accuracies.

    Parameters:
    - name (str): Dataset or experiment name.
    - method_name (str): Name of the attack method.
    - attack_rate (float): Attack rate for labeling the file.
    - ori_changed_edges (int): Number of originally changed edges.
    - filter_changes (list): List of change counts for filter.
    - filter_acc_result (list): List of accuracy results for filter (lists of values for each run).
    - random_acc_result (list): List of accuracy results for random filtering (lists of values for each run).
    - repeat_n (int): Number of repetitions for accuracy tests.
    """
    # Define folder paths and filenames
    output_folder = './evasion_result'
    os.makedirs(output_folder, exist_ok=True)
    file_name = f"{name}_{method_name}_{attack_rate}_{ori_changed_edges}_acc_results_not_retrain.csv"
    file_path = os.path.join(output_folder, file_name)

    # Write data to CSV
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Change'] + [f'Edge_filter_{i + 1}' for i in range(repeat_n)] + \
                 [f'Target_filter_{i + 1}' for i in range(repeat_n)] + [f'Random_filter_{i + 1}' for i in range(repeat_n)]
        writer.writerow(header)
        for i in range(len(filter_changes)):
            row = [filter_changes[i]] + edge_acc_result[i] + filter_acc_result[i] + random_acc_result[i]
            writer.writerow(row)

    # Calculate averages and standard deviations
    filter_avg_values = [np.mean(acc) for acc in filter_acc_result]
    filter_std_values = [np.std(acc) / 2 for acc in filter_acc_result]
    random_avg_values = [np.mean(acc) for acc in random_acc_result]
    random_std_values = [np.std(acc) / 2 for acc in random_acc_result]
    edge_avg_values = [np.mean(acc) for acc in edge_acc_result]
    edge_std_values = [np.std(acc) / 2 for acc in edge_acc_result]

    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.plot(filter_changes, filter_avg_values, label='Filter Avg Accuracy', color='blue', marker='o')
    plt.fill_between(filter_changes,
                     np.array(filter_avg_values) - np.array(filter_std_values),
                     np.array(filter_avg_values) + np.array(filter_std_values),
                     color='blue', alpha=0.2)

    plt.plot(filter_changes, random_avg_values, label='Random Avg Accuracy', color='red', marker='o')
    plt.fill_between(filter_changes,
                     np.array(random_avg_values) - np.array(random_std_values),
                     np.array(random_avg_values) + np.array(random_std_values),
                     color='red', alpha=0.2)

    plt.plot(filter_changes, edge_avg_values, label='Edge-filter Avg Accuracy', color='green', marker='o')
    plt.fill_between(filter_changes,
                     np.array(edge_avg_values) - np.array(edge_std_values),
                     np.array(edge_avg_values) + np.array(edge_std_values),
                     color='green', alpha=0.2)

    plt.xlabel('Changes')
    plt.ylabel('Accuracy')
    plt.title(f'{method_name} Filter vs Random Edge Removal Accuracy for {name}')
    plt.legend()
    plt.grid(True)

    # Save plot as EPS
    output_dir = './evasion_pic_result'
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{source}_{name}_{method_name}_{attack_rate}_{ori_changed_edges}.eps'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, format='eps')

    plt.show()

def get_pseudo_labels(pre_labels, labels, idx_train):
    # 假设 labels 是 numpy.ndarray，predicted_labels 是 Tensor
    predicted_labels = pre_labels.clone()
    predicted_labels = predicted_labels.to('cpu')
    labels = labels.to('cpu')
    predicted_labels[idx_train] = labels[idx_train]
    return predicted_labels