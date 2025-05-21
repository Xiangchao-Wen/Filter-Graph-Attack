from deeprobust.graph.defense import GCN
from utils import *
from gen_attack import generate_modified_adj
import argparse
from scipy.sparse import csr_matrix
from collections import Counter
def get_adjs_acc(model, filter_adj, edge_filter_adj, random_filter_adj, pyg_data, labels,
                        repeat_n, device):
    filter_accs = []
    random_accs = []
    edge_accs = []
    poison_pyg_data = copy.deepcopy(pyg_data)
    for _ in range(repeat_n):
        # 有效性随机移除
        filter_adj_pyg = retype_adj(filter_adj)
        poison_pyg_data.adj_t = filter_adj_pyg
        model.fit(poison_pyg_data, verbose=False)
        preds_poison = model.predict(pyg_data.x, poison_pyg_data.adj_t).argmax(1)
        poison_acc = calculate_accuracy(idx_test, preds_poison, labels)
        filter_accs.append(poison_acc)

        # 有效性边移除
        edge_adj_pyg = retype_adj(edge_filter_adj)
        poison_pyg_data.adj_t = edge_adj_pyg
        model.fit(poison_pyg_data, verbose=False)
        preds_poison = model.predict(pyg_data.x, poison_pyg_data.adj_t).argmax(1)
        poison_acc = calculate_accuracy(idx_test, preds_poison, labels)
        edge_accs.append(poison_acc)

        # 随机移除
        random_adj_pyg = retype_adj(random_filter_adj)
        poison_pyg_data.adj_t = random_adj_pyg
        model.fit(poison_pyg_data, verbose=False)
        preds_poison = model.predict(pyg_data.x, poison_pyg_data.adj_t).argmax(1)
        poison_acc = calculate_accuracy(idx_test, preds_poison, labels)
        random_accs.append(poison_acc)

        # model.fit(features, filter_adj, labels, idx_train,
        #           idx_val)  # train on poison graph with earlystopping
        # acc = gcn_filter.test(idx_test)
        # filter_accs.append(acc)
        # # 有效性边移除
        # gcn_filter = GCN(nfeat=features.shape[1],
        #                  nhid=16,
        #                  nclass=labels.max().item() + 1,
        #                  device=device)
        # gcn_filter = gcn_filter.to(device)
        # gcn_filter.fit(features, edge_filter_adj, labels, idx_train,
        #                idx_val)  # train on poison graph with earlystopping
        # acc = gcn_filter.test(idx_test)
        # edge_accs.append(acc)
        # # 随机移除
        # gcn_random = GCN(nfeat=features.shape[1],
        #                  nhid=16,
        #                  nclass=labels.max().item() + 1,
        #                  device=device)
        # gcn_random = gcn_random.to(device)
        # gcn_random.fit(features, random_filter_adj, labels, idx_train,
        #                idx_val)  # train on clean graph with earlystopping
        # acc = gcn_random.test(idx_test)
        # random_accs.append(acc)
    return filter_accs,edge_accs,random_accs
def get_ensemble_label():
    # 训练原始模型
    clean_model = load_pyg_model(pyg_data, 'gcn', args.source, args.dataset, device, logger, True)
    preds_clean = clean_model.predict(pyg_data.x, pyg_data.adj_t).argmax(1)

    clean_model_appnp = load_pyg_model(pyg_data, 'appnp', args.source, args.dataset, device, logger, True)
    preds_clean_appnp = clean_model_appnp.predict(pyg_data.x, pyg_data.adj_t).argmax(1)

    clean_model_gat = load_pyg_model(pyg_data, 'gat', args.source, args.dataset, device, logger, True)
    preds_clean_gat = clean_model_gat.predict(pyg_data.x, pyg_data.adj_t).argmax(1)

    clean_model_graph_sage = load_pyg_model(pyg_data, 'graph-sage', args.source, args.dataset, device, logger, True)
    preds_clean_graph_sage = clean_model_graph_sage.predict(pyg_data.x, pyg_data.adj_t).argmax(1)

    # 假设 preds_clean_* 都是 shape 为 (num_samples,) 的张量，包含每个模型的预测类别
    preds_clean_gcn = preds_clean.cpu().numpy()
    preds_clean_gat = preds_clean_gat.cpu().numpy()
    preds_clean_graph_sage = preds_clean_graph_sage.cpu().numpy()
    preds_clean_appnp = preds_clean_appnp.cpu().numpy()

    # 将预测结果按列组合起来 (num_samples, num_models)
    all_preds = torch.tensor([
        preds_clean_gcn,
        preds_clean_gat,
        preds_clean_graph_sage,
        preds_clean_appnp
    ]).T  # 转置，使得每一行对应一个样本

    # 多数投票
    ensemble_preds = []
    most_common_counts = []
    for preds in all_preds:
        counts = Counter(preds.tolist())
        most_common_label, most_common_count = counts.most_common(1)[0]  # 获取最多类别及其次数
        # 将结果存储
        ensemble_preds.append(most_common_label)
        if most_common_count ==4:
            most_common_counts.append(0.5)
        else:
            most_common_counts.append(0.05)

    # 将结果转换为张量 (num_samples,)
    ensemble_preds = torch.tensor(ensemble_preds)
    return ensemble_preds, most_common_counts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pubmed', choices=['cora', 'citeseer', 'acm', 'cora_ml', 'polblogs', 'arxiv', 'blogcatalog', 'products','pubmed'])
    parser.add_argument('--attack', type=str, default='Metattack', choices=['prbcd', 'pga', 'greedy','Metattack', 'MinMax', 'PGDAttack','pgdattack-CW', 'DICE', 'DICE_train'])
    parser.add_argument('--source', type=str, default='prognn', choices=['planepoied', 'gcn', 'prognn','ogb'])
    parser.add_argument('--victim', type=str, default='gcn')
    parser.add_argument('--filter_range', type=str, default='all')
    parser.add_argument('--ptb_rate', type=float, default=0.1)
    parser.add_argument('--filter_n', type=int, default=20)
    parser.add_argument('--repeat_n', type=int, default=5)
    parser.add_argument('--select_times', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--use_true_label', type=bool, default=False)
    parser.add_argument('--gpu_id', type=int, default=0)

    epsilon = 0
    break_count = 0
    find_largest = False
    get_break_adj = False
    args = parser.parse_args()
    repeat_n = args.repeat_n
    # print('repeat_n',repeat_n)
    select_times = args.select_times
    device = get_device(args.gpu_id)

    logger_filename = 'models.log'
    logger = get_logger(logger_filename, level=0)

    #
    mod_adj = load_attack_adj(args.source, args.attack, args.dataset, args.ptb_rate)
    mod_adj_pyg = retype_adj(mod_adj)
    # print(type(mod_adj))
    if isinstance(mod_adj, np.ndarray):
        # 将 numpy.ndarray 转换为 csr_matrix
        mod_adj = csr_matrix(mod_adj)

    # print(type(mod_adj))
    # adj, features, labels, idx_train, idx_val, idx_test = load_data(args.source, args.dataset)

    pyg_data = load_pyg_data(args.source, args.dataset)
    edge_index = pyg_data.edge_index
    num_nodes = pyg_data.num_nodes
    adj = sp.csr_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                        shape=(num_nodes, num_nodes))
    features = pyg_data.x.numpy()
    labels = pyg_data.y.numpy()
    idx_train = torch.nonzero(pyg_data.train_mask).squeeze()
    idx_test = torch.nonzero(pyg_data.test_mask).squeeze()
    idx_val = torch.nonzero(pyg_data.val_mask).squeeze()
    print(len(idx_train),len(idx_val),len(idx_test))

    poison_accs = []
    # 训练原始模型
    clean_model = load_pyg_model(pyg_data, args.victim, args.source, args.dataset, device, logger, True)
    preds_clean = clean_model.predict(pyg_data.x, pyg_data.adj_t).argmax(1)

    # 获取伪标签
    if args.use_true_label == False:
        ensemble_preds, confidence_score = get_ensemble_label()
        psudo_labels = get_pseudo_labels(ensemble_preds, pyg_data.y, idx_train)
        pyg_data.y = psudo_labels
        clean_acc = calculate_accuracy(idx_test, ensemble_preds, labels)
        print('ensemble acc', clean_acc)
        print('psudo_label')
    else:
        confidence_score = [1] * len(labels)

    # print(most_common_counts)
    #
    # idx_list_4 = [i for i, val in enumerate(most_common_counts) if val == 4]
    # idx_list_4 = list((set(range(len(labels)))-set(idx_train)) & set(idx_list_4))
    # clean_acc = calculate_accuracy(idx_list_4, ensemble_preds, labels)
    # print('vote4 acc', len(idx_list_4),clean_acc)
    #
    # idx_list_3 = [i for i, val in enumerate(most_common_counts) if val == 3]
    # idx_list_3 = list((set(range(len(labels)))-set(idx_train)) & set(idx_list_3))
    # clean_acc = calculate_accuracy(idx_list_3, ensemble_preds, labels)
    # print('vote3 acc',len(idx_list_3), clean_acc)
    #
    # idx_list_2 = [i for i, val in enumerate(most_common_counts) if val == 2]
    # idx_list_2 = list((set(range(len(labels)))-set(idx_train)) & set(idx_list_2))
    # clean_acc = calculate_accuracy(idx_list_2, ensemble_preds, labels)
    # print('vote2 acc',len(idx_list_2), clean_acc)

    clean_acc = calculate_accuracy(idx_test, preds_clean, labels)
    print('clean acc', clean_acc)
    poison_model = load_pyg_model(pyg_data, args.victim, args.source, args.dataset, device, logger, True)
    poison_pyg_data = copy.deepcopy(pyg_data)
    poison_pyg_data.adj_t = mod_adj_pyg
    for _ in range(repeat_n):
        poison_model.fit(poison_pyg_data, verbose=False)
        preds_poison = poison_model.predict(pyg_data.x, mod_adj_pyg).argmax(1)
        poison_acc = calculate_accuracy(idx_test, preds_poison, labels)
        poison_accs.append(poison_acc)

    average_poison_acc = sum(poison_accs) / len(poison_accs) if len(poison_accs) > 0 else 0
    print('poison acc', average_poison_acc)


    poison_model = load_pyg_model(pyg_data, args.victim, args.source, args.dataset, device, logger, True)
    poison_model.fit(pyg_data, verbose=False)



    # model = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
    #             nhid=256, weight_decay=5e-5, dropout=0.5, with_relu=True, with_bias=True, device=device).to(device)
    # model.fit(features, adj, labels, idx_train)
    # model.test(idx_test)
    #
    # preds_before = model.predict(features, adj).argmax(1)
    # preds_evasion = model.predict(features, mod_adj).argmax(1)
    #
    # poison_accs = []
    # for _ in range(repeat_n):
    #     # poison_model = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
    #     #             nhid=256, weight_decay=5e-5, dropout=0.5, with_relu=True, with_bias=True, device=device).to(device)
    #     poison_model = GCN(nfeat=features.shape[1],
    #                      nhid=16,
    #                      nclass=labels.max().item() + 1,
    #                      device=device).to(device)
    #     # output = model(features, adj)
    #     poison_model.fit(features, mod_adj, labels, idx_train)
    #     poison_acc = poison_model.test(idx_test)
    #     poison_accs.append(poison_acc)
    #
    # # 调用函数
    # ori_acc = calculate_accuracy(idx_test, preds_before, labels)
    # attack_acc = calculate_accuracy(idx_test, preds_evasion, labels)
    # # print("干净邻接矩阵的最大值:", adj.data.max())
    # # print("污染邻接矩阵的最大值:", mod_adj.data.max())
    # print('ori acc1', ori_acc)
    # print('evasion acc1', attack_acc)
    # print('poison acc',poison_acc)

    ori_changed_edges = change_num(adj, mod_adj)
    print('edge_change',ori_changed_edges)
    """选取并集"""
    # selected_idx_test = []
    # for victim_name in ['gcn', 'sgc', 'gat', 'graph-sage']:
    #     victim = load_pyg_model(pyg_data, victim_name, args.source, args.dataset, device, logger, True)
    #     selected_idx_test += filter_poison_pyg(pyg_data, mod_adj_pyg, idx_test, repeat_n, victim)

    # selected_idx_test = filter_poison(adj, mod_adj, features, idx_train, idx_val, idx_test, labels, 6, device)

    if args.filter_range == 'all':
        final_filter_adj = adj
    else:
        """选取交集"""
        # 创建一个 Counter 对象来记录每个元素出现的次数
        selected_idx_test = []
        idx_counter = Counter()

        # 遍历每个 victim_name
        name1 = ['gcn', 'sgc', 'gat', 'graph-sage']

        for victim_name in ['gcn', 'sgc', 'gat', 'graph-sage']:
            victim = load_pyg_model(pyg_data, victim_name, args.source, args.dataset, device, logger, True)
            if args.attack=='Metattack' or args.attack=='MinMax':
                victim_selected_idx = filter_poison_pyg(pyg_data, mod_adj_pyg, 1, victim, filter_method='train')
            elif args.attack=='prbcd':
                victim_selected_idx = filter_poison_pyg(pyg_data, mod_adj_pyg, 1, victim, filter_method='train')
            else:
                victim_selected_idx = filter_poison_pyg(pyg_data, mod_adj_pyg, 1, victim, filter_method='train')

            # 将当前 victim_selected_idx 中的元素添加到 Counter 中
            idx_counter.update(victim_selected_idx)

        # 筛选出出现次数大于等于2的元素（即至少出现在两个集合中）
        # select_times越大，有效节点越少，无用边候选集就越大
        selected_idx_test = [idx for idx, count in idx_counter.items() if count >= select_times]

        final_filter_adj = get_new_adj(adj, mod_adj, selected_idx_test)

    filter_changed_edges = change_num(adj, final_filter_adj)
    total_filter_edges = ori_changed_edges - filter_changed_edges
    pre_filter_num = int(total_filter_edges / args.filter_n)
    filter_N = args.filter_n
    print('filter_edge', total_filter_edges, pre_filter_num)
    if pre_filter_num < 1:
        os._exit(0)


    print('change edge', total_filter_edges)
    # for i in range(5):
    #     mod_adj_pyg = retype_adj(final_filter_adj)
    #     poison_model = load_pyg_model(pyg_data, 'gcn', args.source, args.dataset, device, logger, True)
    #     poison_pyg_data = copy.deepcopy(pyg_data)
    #     poison_pyg_data.adj_t = mod_adj_pyg
    #     poison_model.fit(poison_pyg_data, verbose=False)
    #     preds_poison = poison_model.predict(pyg_data.x, mod_adj_pyg).argmax(1)
    #     poison_acc = calculate_accuracy(idx_test, preds_poison, labels)
    #     print('target filter acc', poison_acc)
    #
    #     mod_adj_pyg = retype_adj(final_random_adj)
    #     poison_model = load_pyg_model(pyg_data, 'gcn', args.source, args.dataset, device, logger, True)
    #     poison_pyg_data = copy.deepcopy(pyg_data)
    #     poison_pyg_data.adj_t = mod_adj_pyg
    #     poison_model.fit(poison_pyg_data, verbose=False)
    #     preds_poison = poison_model.predict(pyg_data.x, mod_adj_pyg).argmax(1)
    #     poison_acc = calculate_accuracy(idx_test, preds_poison, labels)
    #     print('random filter acc', poison_acc)

    target_filter_accs = []
    random_filter_accs = []

    for i in range(repeat_n):
        # 处理 target filter 的部分
        mod_adj_pyg = retype_adj(final_filter_adj)
        poison_model1 = load_pyg_model(pyg_data, args.victim, args.source, args.dataset, device, logger, True)
        poison_pyg_data = copy.deepcopy(pyg_data)
        poison_pyg_data.adj_t = mod_adj_pyg
        poison_model1.fit(poison_pyg_data, verbose=False)
        preds_poison = poison_model1.predict(pyg_data.x, mod_adj_pyg).argmax(1)
        poison_acc = calculate_accuracy(idx_test, preds_poison, labels)
        target_filter_accs.append(poison_acc)

        # 处理 random filter 的部分
        final_random_adj = toggle_attack_edges(adj, mod_adj, filter_num=total_filter_edges)
        filter_changed_edges = change_num(adj, final_random_adj)
        print('random filter edge change',filter_changed_edges)
        mod_adj_pyg = retype_adj(final_random_adj)
        poison_model2 = load_pyg_model(pyg_data, args.victim, args.source, args.dataset, device, logger, True)
        poison_pyg_data = copy.deepcopy(pyg_data)
        poison_pyg_data.adj_t = mod_adj_pyg
        poison_model2.fit(poison_pyg_data, verbose=False)
        preds_poison = poison_model2.predict(pyg_data.x, mod_adj_pyg).argmax(1)
        poison_acc = calculate_accuracy(idx_test, preds_poison, labels)
        random_filter_accs.append(poison_acc)

    # 计算并输出平均值
    average_target_filter_acc = sum(target_filter_accs) / len(target_filter_accs)
    average_random_filter_acc = sum(random_filter_accs) / len(random_filter_accs)


    print('Average target filter accuracy:', average_target_filter_acc)
    print('Average random filter accuracy:', average_random_filter_acc)

    # for i in range(30):
    #     filters = int((i+1)*(ori_changed_edges/30))
    #     final_random_adj = toggle_attack_edges(adj, mod_adj, filter_num=filters)
    #     mod_adj_pyg = retype_adj(final_random_adj)
    #     poison_model = load_pyg_model(pyg_data, 'gcn', args.source, args.dataset, device, logger, True)
    #     poison_pyg_data = copy.deepcopy(pyg_data)
    #     poison_pyg_data.adj_t = mod_adj_pyg
    #     poison_model.fit(poison_pyg_data, verbose=False)
    #     preds_poison = poison_model.predict(pyg_data.x, mod_adj_pyg).argmax(1)
    #     poison_acc = calculate_accuracy(idx_test, preds_poison, labels)
    #     print(filters,poison_acc)
    # os._exit(0)

    random_acc_result = []
    filter_acc_result = []
    edge_acc_result = []
    random_acc_result.append(poison_accs)
    filter_acc_result.append(poison_accs)
    edge_acc_result.append(poison_accs)

    filter_changes = []
    filter_changes.append(0)
    filter_adj = mod_adj.copy()
    random_filter_adj = mod_adj.copy()
    edge_filter_adj = mod_adj.copy()

    for i in range(filter_N + 1):
        if (i != filter_N):
            if epsilon == 0:
                # edge_filter_adj, epsilon, search_count, break_adj = filter_edge_combined(model, poison_model, final_filter_adj, edge_filter_adj,
                #                                                 idx_train, idx_test,
                #                                                 features,
                #                                                 labels, filter_num=pre_filter_num, epsilon=epsilon, gpu_id=args.gpu_id)
                edge_filter_adj, epsilon, search_count, break_adj = filter_edge_combined_pyg_batch(clean_model, poison_model, final_filter_adj, edge_filter_adj,
                                                                                                pyg_data, pre_filter_num, epsilon, gpu_id=args.gpu_id, batch_size=args.batch_size,
                                                                                                confidence_score=confidence_score)
                perturb_adj_pyg = retype_adj(edge_filter_adj)
                logits_poison = poison_model.predict(pyg_data.x, perturb_adj_pyg)
                criterion = nn.CrossEntropyLoss(reduction='none')
                labels_tensor = torch.tensor(labels, dtype=torch.long).cuda(args.gpu_id)
                loss_poison = criterion(logits_poison, labels_tensor).cpu().detach().numpy()
                avg_loss_poison_train = np.mean(loss_poison[idx_train])
                avg_loss_poison_test = np.mean(loss_poison[idx_test])
                print(avg_loss_poison_test,avg_loss_poison_train)
            else:
                edge_filter_adj = toggle_attack_edges(final_filter_adj, edge_filter_adj, filter_num=pre_filter_num)

            if find_largest==False and search_count >= 0:
                break_count = search_count + i * pre_filter_num
                find_largest = True
                get_break_adj = True
                searched_adj = break_adj
                filter_break_adj = toggle_attack_edges(final_filter_adj, filter_adj, filter_num=search_count)
                random_filter_break_adj = toggle_attack_edges(adj, random_filter_adj, filter_num=search_count)

            filter_adj = toggle_attack_edges(final_filter_adj, filter_adj, filter_num=pre_filter_num)
            random_filter_adj = toggle_attack_edges(adj, random_filter_adj, filter_num=pre_filter_num)
        else:
            if find_largest==False:
                break_count = i * pre_filter_num
                searched_adj = edge_filter_adj

            filter_adj = final_filter_adj.copy()
            edge_filter_adj = final_filter_adj.copy()
            random_filter_adj = toggle_attack_edges(adj, random_filter_adj,
                                                    filter_num=filter_edges - pre_filter_num * filter_N)
        if get_break_adj==True:
            filter_accs, edge_accs, random_accs = get_adjs_acc(clean_model, filter_break_adj, searched_adj, random_filter_break_adj, pyg_data,
                                                               labels, repeat_n, device)
            filter_changes.append(break_count)
            filter_acc_result.append(filter_accs)
            random_acc_result.append(random_accs)
            edge_acc_result.append(edge_accs)
            get_break_adj = False

        filter_changed_edges = change_num(adj, filter_adj)
        edge_filter_changed_edges = change_num(adj, edge_filter_adj)
        random_changed_edges = change_num(adj, random_filter_adj)
        filter_edges = ori_changed_edges - filter_changed_edges
        random_filter_edges = ori_changed_edges - random_changed_edges
        edge_filter_edges = ori_changed_edges - edge_filter_changed_edges
        # filter_accs,edge_accs,random_accs = get_adjs_acc(filter_adj, edge_filter_adj, random_filter_adj, features, labels, idx_train,
        #              idx_val, repeat_n,device)
        filter_accs,edge_accs,random_accs = get_adjs_acc(clean_model, filter_adj, edge_filter_adj, random_filter_adj, pyg_data, labels,
                     repeat_n, device)
        filter_changes.append(filter_edges)
        filter_acc_result.append(filter_accs)
        random_acc_result.append(random_accs)
        edge_acc_result.append(edge_accs)

        print('total attack edge', ori_changed_edges,'filter edges', filter_edges)
        # print('random filter edges', random_filter_edges)
        print('Average accuracy on edge filter graph:', np.mean(edge_accs))
        print('Average accuracy on filter graph:', np.mean(filter_accs))
        print('Average accuracy on random filter graph:', np.mean(random_accs))
        # filter_adj = filter_edge(victim_model, gcn_filter, adj, filter_adj, features, pseudo_labels, filter_num=10, alpha=0)

        if find_largest == True:
            break


    break_changed_edges = change_num(adj, searched_adj)
    break_attack_rate = np.round(break_changed_edges / (pyg_data.num_edges // 2), 4)
    print('attack_rate', break_attack_rate)
    mod_adj_pyg = retype_adj(searched_adj)
    poison_model = load_pyg_model(pyg_data, 'gcn', args.source, args.dataset, device, logger, True)
    poison_pyg_data = copy.deepcopy(pyg_data)
    poison_pyg_data.adj_t = mod_adj_pyg
    poison_model.fit(poison_pyg_data, verbose=False)
    preds_poison = poison_model.predict(pyg_data.x, mod_adj_pyg).argmax(1)
    poison_acc = calculate_accuracy(idx_test, preds_poison, labels)

    print('change edge',break_changed_edges)
    print('filter poison acc',poison_acc)



    # args.source, args.attack, args.dataset, args.ptb_rate
    save_results_poison(args.source, args.dataset, args.attack, args.ptb_rate, ori_changed_edges, filter_changes, edge_acc_result, filter_acc_result,
                          random_acc_result, break_count, repeat_n)

    break_attack_rate = np.round((ori_changed_edges - break_count) / (pyg_data.num_edges // 2), 4)
    allfilter_attack_rate = np.round((ori_changed_edges - total_filter_edges) / (pyg_data.num_edges // 2), 4)
    save_adj_to_npz(args.source, args.attack, args.dataset, args.ptb_rate, searched_adj, 'break_adj', break_attack_rate)
    save_adj_to_npz(args.source, args.attack, args.dataset, args.ptb_rate, final_filter_adj, 'final_adj', allfilter_attack_rate)
    save_adj_to_npz(args.source, args.attack, args.dataset, args.ptb_rate, random_filter_adj, 'random_adj',
                    allfilter_attack_rate)
    os._exit(0)

