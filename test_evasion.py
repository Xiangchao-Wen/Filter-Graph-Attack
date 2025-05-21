from deeprobust.graph.defense import GCN
from utils import *
from gen_attack import generate_modified_adj
import argparse
from scipy.sparse import csr_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='acm', choices=['cora', 'citeseer', 'acm', 'cora_ml', 'polblogs','blogcatalog','arxiv','products'])
    parser.add_argument('--attack', type=str, default='prbcd', choices=['prbcd', 'apga', 'pga', 'greedy','Metattack', 'MinMax', 'PGDAttack','pgdattack-CW', 'DICE', 'DICE_train', 'random'])
    parser.add_argument('--victim', type=str, default='gcn', choices=['gcn', 'sgc', 'gat', 'graph-sage'])
    parser.add_argument('--source', type=str, default='gcn', choices=['planepoied', 'gcn', 'prognn','ogb'])
    parser.add_argument('--ptb_rate', type=float, default=0.25)
    parser.add_argument('--repeat_n', type=int, default=4)
    parser.add_argument('--select_times', type=int, default=1)
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--gpu_id', type=int, default=7)

    # parser.add_argument('--filter_n', type=int, default=20)

    args = parser.parse_args()
    repeat_n = args.repeat_n
    select_times = args.select_times
    # device = 'cuda'
    device = get_device(args.gpu_id)

    logger_filename = 'evasion_attack-' + args.dataset + '-' + args.attack + '.log'
    logger_name = 'evaluate'
    logger = get_logger(logger_filename, level=0, name=logger_name)


    models = [
    'gcn', 'sgc', 'gat', 'graph-sage', 'rgcn', 'median-gcn', 'gcn-jaccard',
    'gnn-guard', 'grand', 'appnp'
    ]
    # models = ['gcn']
    # models = ['graph-sage']

    mod_adj = load_attack_adj(args.source, args.attack, args.dataset, args.ptb_rate)
    # print(type(mod_adj))
    if isinstance(mod_adj, np.ndarray):
        # 将 numpy.ndarray 转换为 csr_matrix
        mod_adj = csr_matrix(mod_adj)
    print(type(mod_adj))
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.source, args.dataset)
    # print(len(idx_train),len(idx_test))
    pyg_data = load_pyg_data(args.source,args.dataset)


    """
    idx_train = torch.nonzero(pyg_data.train_mask).squeeze()
    idx_test = torch.nonzero(pyg_data.test_mask).squeeze()
    idx_val = torch.nonzero(pyg_data.val_mask).squeeze()
    print(len(idx_train), len(idx_test))

    features2 = pyg_data.x.numpy()


    # 检查形状是否一致
    if features.shape != features2.shape:
        print("矩阵形状不同，无法比较")
    else:
        # 逐元素差异
        diff = features - features2
        print("逐元素差异矩阵：\n", diff)

        # 均方误差
        if isinstance(diff, np.matrix):
            diff = np.asarray(diff)

        mse = np.mean(diff ** 2)
        print("均方误差：", mse)

        # 是否完全相等
        are_equal = np.array_equal(features, features2)
        print("矩阵是否相等：", are_equal)

        # 统计不同元素个数
        diff_count = np.sum(features != features2)
        print("不同元素的个数：", diff_count)


    labels = pyg_data.y.numpy()
    adj_pyg = pyg_data.adj_t

    row, col, value = adj_pyg.coo()
    if value is None:
        value = torch.ones(row.size(0), dtype=torch.float32)

    # 将其转换为 scipy.sparse.csr_matrix
    csr_mat = csr_matrix((value.numpy(), (row.numpy(), col.numpy())), shape=adj_pyg.sizes())

    # changes = change_num(adj, csr_mat)
    # print(changes)
    """

    """
    # 测试
    model = load_pyg_model(pyg_data, models[0], args.source, args.dataset, device, logger, args.pretrained)

    preds_before = model.predict(pyg_data.x, pyg_data.adj_t).argmax(1)
    # print(type(pyg_data.adj_t))

    mod_adj_pyg = retype_adj(mod_adj)
    preds_evasion = model.predict(pyg_data.x, mod_adj_pyg).argmax(1)

    ori_acc = calculate_accuracy(idx_test, preds_before, labels)
    attack_acc = calculate_accuracy(idx_test, preds_evasion, labels)
    print('ori,eva',ori_acc,attack_acc)

    model = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                nhid=256, weight_decay=5e-5, dropout=0.5, with_relu=True, with_bias=True, device='cuda').to('cuda')
    model.fit(features, csr_mat, labels, idx_train)
    model.test(idx_test)

    preds_before = model.predict(features, csr_mat).argmax(1)
    preds_evasion = model.predict(features, mod_adj).argmax(1)
    ori_acc = calculate_accuracy(idx_test, preds_before, labels)
    attack_acc = calculate_accuracy(idx_test, preds_evasion, labels)

    print('ori acc1', ori_acc)
    print('evasion acc1', attack_acc)
    #测试结束
    """




    # 原始扰动数量
    ori_changed_edges = change_num(adj, mod_adj)
    print('change n',ori_changed_edges)

    # 筛选有效节点
    # pyg_data, mod_adj, idx_test, repeat_n, victim
    selected_idx_test = []
    for victim_name in ['gcn', 'sgc', 'gat', 'graph-sage']:
        victim = load_pyg_model(pyg_data,victim_name,args.source,args.dataset,device,logger,True)
        selected_idx_test += filter_evasion_pyg(pyg_data,mod_adj,idx_test, select_times, victim)
    final_filter_adj = get_new_adj(adj, mod_adj, selected_idx_test)
    filter_changed_edges = change_num(adj, final_filter_adj)
    # 过滤扰动数量
    total_filter_edges = ori_changed_edges - filter_changed_edges
    print('filter_n',total_filter_edges)
    random_filter_adj = mod_adj.copy()
    random_filter_adj = toggle_attack_edges(adj, random_filter_adj, filter_num=total_filter_edges)

    # 初始化存储结果的列表
    results = []
    for name in models:
        print(name)
        ori_accs, attack_accs, filter_accs, random_accs = [], [], [], []
        for _ in range(repeat_n):
            print(_,repeat_n)
            if (name == 'median-gcn' or name == 'grand') and args.dataset=='blogcatalog':
                model = load_pyg_model(pyg_data, name, args.source, args.dataset, device, logger, True)
            else:
                model = load_pyg_model(pyg_data, name, args.source, args.dataset, device, logger, args.pretrained)

            preds_before = model.predict(pyg_data.x, pyg_data.adj_t).argmax(1)
            # print(type(pyg_data.adj_t))

            mod_adj_pyg = retype_adj(mod_adj)
            preds_evasion = model.predict(pyg_data.x, mod_adj_pyg).argmax(1)

            ori_acc = calculate_accuracy(idx_test, preds_before, labels)
            attack_acc = calculate_accuracy(idx_test, preds_evasion, labels)

            final_filter_adj_pyg = retype_adj(final_filter_adj)
            preds_filter = model.predict(pyg_data.x, final_filter_adj_pyg).argmax(1)

            random_filter_adj_pyg = retype_adj(random_filter_adj)
            preds_random = model.predict(pyg_data.x, random_filter_adj_pyg).argmax(1)
            filter_acc = calculate_accuracy(idx_test, preds_filter, labels)
            random_acc = calculate_accuracy(idx_test, preds_random, labels)

            # 记录结果
            ori_accs.append(ori_acc)
            attack_accs.append(attack_acc)
            filter_accs.append(filter_acc)
            random_accs.append(random_acc)
        # 统计均值和方差
        result = {
            "model": name,
            "filter_changed_edges": filter_changed_edges,
            "total_filter_edges": total_filter_edges,
            "ori_acc_mean": sum(ori_accs) / repeat_n,
            "ori_acc_std": (sum((x - sum(ori_accs) / repeat_n) ** 2 for x in ori_accs) / repeat_n) ** 0.5,
            "attack_acc_mean": sum(attack_accs) / repeat_n,
            "attack_acc_std": (sum((x - sum(attack_accs) / repeat_n) ** 2 for x in attack_accs) / repeat_n) ** 0.5,
            "filter_acc_mean": sum(filter_accs) / repeat_n,
            "filter_acc_std": (sum((x - sum(filter_accs) / repeat_n) ** 2 for x in filter_accs) / repeat_n) ** 0.5,
            "random_acc_mean": sum(random_accs) / repeat_n,
            "random_acc_std": (sum((x - sum(random_accs) / repeat_n) ** 2 for x in random_accs) / repeat_n) ** 0.5,
        }
        # 添加到结果列表
        results.append(result)

    # 转为 DataFrame 并保存为表格
    df = pd.DataFrame(results)

    output_folder = f'./evasion_result/{args.source}'
    os.makedirs(output_folder, exist_ok=True)
    file_name = f"{args.dataset}_{args.attack}_{args.ptb_rate}_{ori_changed_edges}.csv"
    file_path = os.path.join(output_folder, file_name)

    df.to_csv(file_path, index=False)
    print(file_path,'has saved')

    # args.source, args.attack, args.dataset, args.ptb_rate
    # save_results_evasion(args.source, args.dataset, args.attack, args.ptb_rate, ori_changed_edges,
    #                      ori_acc_list, attack_acc_list, filter_acc_list, random_acc_list,
    #                      filter_changed_edges_list)

    save_adj_to_npz(args.source, args.attack, args.dataset, args.ptb_rate, final_filter_adj, 'evasion_filter_adj')
    # import os

    os._exit(0)



    # def save_results_evasion(source, name, method_name, attack_rate, ori_changed_edges,
    #                          ori_acc_list, attack_acc_list, filter_acc_list, random_acc_list,
    #                          filter_changed_edges_list):