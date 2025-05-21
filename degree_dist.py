from utils import *
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wasserstein_distance
from scipy.stats import chisquare
from scipy.optimize import minimize_scalar
from scipy.stats import chi2
import numpy as np
from scipy.optimize import minimize


def compute_lambda_statistic(adj_degrees, mod_adj_degrees, d_min=1):
    # 过滤度数以保留不小于d_min的节点
    D_G0 = adj_degrees[adj_degrees >= d_min]
    D_G_prime = mod_adj_degrees[mod_adj_degrees >= d_min]
    D_comb = np.concatenate([D_G0, D_G_prime])

    # 计算各数据集的alpha参数
    def compute_alpha(D, d_min):
        if len(D) < 1:
            return 0.0
        d_min_half = d_min - 0.5
        sum_log_terms = np.sum(np.log(D / d_min_half))
        return 1 + len(D) / sum_log_terms

    alpha_G0 = compute_alpha(D_G0, d_min)
    alpha_G_prime = compute_alpha(D_G_prime, d_min)
    alpha_comb = compute_alpha(D_comb, d_min)

    # 计算对数似然函数
    def log_likelihood(D, alpha, d_min):
        if len(D) < 1 or alpha <= 0:
            return 0.0
        n = len(D)
        term1 = n * np.log(alpha)
        term2 = alpha * n * np.log(d_min)
        term3 = (alpha + 1) * np.sum(np.log(D))
        return term1 + term2 - term3

    # 计算假设检验统计量
    l_h0 = log_likelihood(D_comb, alpha_comb, d_min)
    l_h1 = log_likelihood(D_G0, alpha_G0, d_min) + log_likelihood(D_G_prime, alpha_G_prime, d_min)

    # 计算最终的Lambda统计量
    Lambda = -2 * l_h0 + 2 * l_h1
    return Lambda
def compute_powerlaw_Lambda(adj_degrees, mod_adj_degrees, dmin=1):
    def filter_degrees(degrees, dmin):
        return degrees[degrees >= dmin]

    def estimate_alpha(deg, dmin):
        n = len(deg)
        sum_log = np.sum(np.log(deg / (dmin - 0.5)))
        return 1 + n / sum_log

    def log_likelihood(deg, alpha, dmin):
        n = len(deg)
        term1 = n * np.log(alpha)
        term2 = n * alpha * np.log(dmin)
        term3 = (alpha + 1) * np.sum(np.log(deg))
        return term1 + term2 - term3

    deg_g0 = filter_degrees(adj_degrees, dmin)
    deg_g1 = filter_degrees(mod_adj_degrees, dmin)
    deg_comb = np.concatenate([deg_g0, deg_g1])

    alpha_g0 = estimate_alpha(deg_g0, dmin)
    alpha_g1 = estimate_alpha(deg_g1, dmin)
    alpha_comb = estimate_alpha(deg_comb, dmin)

    ll_g0 = log_likelihood(deg_g0, alpha_g0, dmin)
    ll_g1 = log_likelihood(deg_g1, alpha_g1, dmin)
    ll_comb = log_likelihood(deg_comb, alpha_comb, dmin)

    Lambda = -2 * ll_comb + 2 * (ll_g0 + ll_g1)
    return Lambda

# def log_likelihood(data, alpha, dmin):
#     n = len(data)
#     sum_log = np.sum(np.log(data))
#     return n * np.log(alpha) + n * alpha * np.log(dmin) - (alpha + 1) * sum_log
#
# def estimate_alpha(data, dmin):
#     def neg_log_likelihood(alpha):
#         return -log_likelihood(data, alpha, dmin)
#     res = minimize_scalar(neg_log_likelihood, bounds=(1.01, 5.0), method='bounded')
#     return res.x
#
# def powerlaw_likelihood_ratio_test(degrees_1, degrees_2, dmin=2):
#     # 筛选符合条件的度数
#     d0 = degrees_1[degrees_1 >= dmin]
#     d1 = degrees_2[degrees_2 >= dmin]
#     d_comb = np.concatenate([d0, d1])
#
#     # 拟合 alpha 参数
#     alpha0 = estimate_alpha(d0, dmin)
#     alpha1 = estimate_alpha(d1, dmin)
#     alpha_comb = estimate_alpha(d_comb, dmin)
#
#     # 计算对数似然
#     l0 = log_likelihood(d0, alpha0, dmin)
#     l1 = log_likelihood(d1, alpha1, dmin)
#     l_comb = log_likelihood(d_comb, alpha_comb, dmin)
#
#     # 构造检验统计量 Λ 和 p-value
#     Lambda = -2 * l_comb + 2 * (l0 + l1)
#     p_value = chi2.sf(Lambda, df=1)
#
#     return Lambda, p_value


def estimate_alpha(degrees, d_min):
    """根据公式 (6) 估计幂律分布的缩放参数 alpha"""
    n = len(degrees)
    sum_log = np.sum(np.log(degrees / (d_min - 0.5)))
    alpha = 1 + n / sum_log
    return alpha


def log_likelihood(degrees, alpha, d_min):
    """计算公式 (7) 的对数似然值 l(D_x)"""
    n = len(degrees)
    term1 = n * np.log(alpha)
    term2 = n * alpha * np.log(d_min)
    term3 = (alpha + 1) * np.sum(np.log(degrees))
    return term1 + term2 - term3


def compute_lambda(adj_degrees, mod_adj_degrees, d_min=1e-5):
    """计算 Λ(G(0), G′) 并检查是否满足扰动条件"""
    # 如果未指定 d_min，取最小度数
    if d_min is None:
        d_min = min(np.min(adj_degrees), np.min(mod_adj_degrees))

    # 过滤度数 < d_min 的节点
    adj_degrees = adj_degrees[adj_degrees >= d_min]
    mod_adj_degrees = mod_adj_degrees[mod_adj_degrees >= d_min]
    if len(adj_degrees) == 0 or len(mod_adj_degrees) == 0:
        raise ValueError("No degrees >= d_min in one or both degree sets.")

    # 估计 alpha
    alpha_G = estimate_alpha(adj_degrees, d_min)
    alpha_G_prime = estimate_alpha(mod_adj_degrees, d_min)

    # 合并样本 D_comb
    combined_degrees = np.concatenate([adj_degrees, mod_adj_degrees])
    alpha_comb = estimate_alpha(combined_degrees, d_min)

    # 计算 l(H0) = l(D_comb)
    l_H0 = log_likelihood(combined_degrees, alpha_comb, d_min)

    # 计算 l(H1) = l(D_G(0)) + l(D_G′)
    l_G = log_likelihood(adj_degrees, alpha_G, d_min)
    l_G_prime = log_likelihood(mod_adj_degrees, alpha_G_prime, d_min)
    l_H1 = l_G + l_G_prime

    # 计算 Λ = -2 * l(H0) + 2 * l(H1)
    Lambda = -2 * l_H0 + 2 * l_H1

    # 检查是否满足扰动条件
    tau = chi2.ppf(0.05, df=1)  # p-value = 0.95，约 0.003932
    is_accepted = Lambda < tau

    return Lambda, is_accepted, tau



dataset='polblogs'
attack = 'greedy'
# filter_break_adj = sp.load_npz('./filter_data/deeprobust/gcn/adj/Metattack-polblogs-0.25-break_adj-0.0857.npz')
filter_break_adj = sp.load_npz('./filter_data/deeprobust/gcn/adj_old/greedy-polblogs-0.25-evasion_filter_adj.npz')
mod_adj = load_attack_adj('gcn',attack, dataset, 0.25)
adj, features, labels, idx_train, idx_val, idx_test = load_data('gcn',dataset)
pyg_data = load_pyg_data('gcn',dataset)
logger_filename = 'evasion_attack.log'
logger_name = 'evaluate'
logger = get_logger(logger_filename, level=0, name=logger_name)
# model = load_pyg_model(pyg_data,'gcn','gcn',dataset,'cpu',logger,False)
# preds_before = model.predict(pyg_data.x, pyg_data.adj_t).argmax(1)
# test_mask = pyg_data.test_mask  # 或者你手动传入的 mask
# test_labels = pyg_data.y[test_mask]
# test_preds = preds_before[test_mask]
#
# accuracy = (test_preds == test_labels).float().mean().item()
#
# print(f"Test Accuracy: {accuracy * 100:.2f}%")
def plot_degree():
    # 计算节点度
    def get_degrees(sparse_adj):
        return np.array(sparse_adj.sum(axis=1)).flatten()

    adj_degrees = get_degrees(adj)
    mod_adj_degrees = get_degrees(mod_adj)
    filter_break_adj_degrees = get_degrees(filter_break_adj)

    min1 = min(adj_degrees)
    min2 = min(mod_adj_degrees)
    min3 = min(filter_break_adj_degrees)
    print(min1,min2,min3)

    # print(len(adj_degrees), adj_degrees)
    # print(len(mod_adj_degrees), mod_adj_degrees)
    # print(len(mod_adj_degrees), mod_adj_degrees)

    Lambda = compute_lambda(adj_degrees, mod_adj_degrees)
    print(f"Λ(G(0), G′) = {Lambda}")
    Lambda = compute_lambda(adj_degrees, filter_break_adj_degrees)
    print(f"Λ(G(0), G′) = {Lambda}")

    Lambda_val = compute_powerlaw_Lambda(adj_degrees, mod_adj_degrees)
    print(f"Lambda: {Lambda_val}")

    Lambda_val = compute_powerlaw_Lambda(adj_degrees, filter_break_adj_degrees)
    print(f"Lambda: {Lambda_val}")

    lambda_value = compute_lambda_statistic(adj_degrees, mod_adj_degrees, d_min=1)
    print(f"New Lambda: {lambda_value}")

    lambda_value = compute_lambda_statistic(adj_degrees, filter_break_adj_degrees, d_min=1)
    print(f"New Lambda: {lambda_value}")
    # Lambda, p_value = powerlaw_likelihood_ratio_test(adj_degrees, mod_adj_degrees)
    #
    # print("Λ =", Lambda)
    # print("p-value =", p_value)
    #
    # Lambda, p_value = powerlaw_likelihood_ratio_test(adj_degrees, filter_break_adj_degrees)
    #
    # print("Λ =", Lambda)
    # print("p-value =", p_value)

    # # Wasserstein 距离
    # dist1 = wasserstein_distance(adj_degrees, mod_adj_degrees)
    # print('原始攻击和原始图度分布Wasserstein距离',dist1)
    # dist1 = wasserstein_distance(adj_degrees, filter_break_adj_degrees)
    # print('过滤攻击和原始图度分布Wasserstein距离',dist1)
    # dist2 = wasserstein_distance(mod_adj_degrees, filter_break_adj_degrees)
    # print('过滤攻击和原始攻击度分布Wasserstein距离', dist2)
    # 绘图
    # plt.figure(figsize=(8, 5))
    # bins = np.arange(0, max(adj_degrees.max(), mod_adj_degrees.max(), filter_break_adj_degrees.max()) + 2) - 0.5
    # plt.hist(adj_degrees, bins=bins, alpha=0.6, label='Original Graph', color='blue')
    # plt.hist(mod_adj_degrees, bins=bins, alpha=0.6, label='Original Attack', color='green')
    # plt.hist(filter_break_adj_degrees, bins=bins, alpha=0.6, label='Filtered Attack', color='red')
    #
    # plt.xlabel("Node Degree")
    # plt.ylabel("Frequency")
    # plt.title("Node Degree Distributions")
    # plt.legend()
    # plt.grid(True)
    #
    # # 保存为 PDF
    # pdf_path = "./evasion_result/prbcd_cora_distributions_evasion.pdf"
    # with PdfPages(pdf_path) as pdf:
    #     pdf.savefig()
    #     plt.close()
plot_degree()
def plot_robutness(mod_adj):
    preds_before = model.predict(pyg_data.x, pyg_data.adj_t).argmax(1)
    # print(type(pyg_data.adj_t))

    mod_adj_pyg = retype_adj(mod_adj)
    preds_evasion = model.predict(pyg_data.x, mod_adj_pyg).argmax(1)

    # Step 2: 找到扰动节点（mod_adj != original adj）
    orig_adj = pyg_data.adj_t.to_scipy(layout="csr").astype(bool)
    mod_adj_bool = mod_adj.astype(bool)

    diff_matrix = mod_adj_bool != orig_adj
    row_diff, col_diff = diff_matrix.nonzero()
    perturbed_nodes = set(row_diff.tolist() + col_diff.tolist())
    perturbed_nodes = torch.tensor(list(perturbed_nodes), dtype=torch.long)

    # Step 3: 分类节点：预测是否改变
    changed_mask = preds_before[perturbed_nodes] != preds_evasion[perturbed_nodes]
    unchanged_mask = ~changed_mask

    changed_nodes = perturbed_nodes[changed_mask]
    unchanged_nodes = perturbed_nodes[unchanged_mask]

    # Step 4: 获取这些节点在原始图中的度
    degrees = np.array(orig_adj.sum(axis=1)).flatten()

    changed_degrees = degrees[changed_nodes.numpy()]
    unchanged_degrees = degrees[unchanged_nodes.numpy()]

    avg_changed = changed_degrees.mean() if len(changed_degrees) > 0 else 0
    avg_unchanged = unchanged_degrees.mean() if len(unchanged_degrees) > 0 else 0

    # Step 5: 绘图
    plt.figure(figsize=(6, 4))
    plt.bar(['Changed', 'Unchanged'], [avg_changed, avg_unchanged], color=['tomato', 'skyblue'])
    plt.ylabel('Average Degree (Original Graph)')
    plt.title(f'Avg Degree of Perturbed Nodes ({dataset}, {attack})')
    plt.tight_layout()

    # Step 6: 保存为 PDF
    filename = f'./evasion_result/avg_degree_changed_{dataset}_{attack}.pdf'
    plt.savefig(filename)
    print(f"Saved to {filename}")
def plot_all_degree():
    result = {
        'cora':[3.11,4.78],
        'cora-ml':[4.01,7],
        'citeseer':[2.12,3],
        'acm':[4.5,8.5],
        'polblogs':[5.1,42.3],
        'blogcatalog':[55,70]
    }

    datasets = list(result.keys())
    weak_vals = [v[0] for v in result.values()]
    stable_vals = [v[1] for v in result.values()]

    x = np.arange(len(datasets))
    width = 0.35

    # 设置论文风格
    plt.rcParams.update({
        "font.size": 12,
        "pdf.fonttype": 42,  # 解决论文中字体显示问题
        "ps.fonttype": 42,
        "figure.figsize": (10, 5),
    })

    # 绘图
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/2, weak_vals, width, label='Weak Node', color='#1b9e77')
    bar2 = ax.bar(x + width/2, stable_vals, width, label='Stable Node', color='#d95f02')

    # 坐标轴和标签
    ax.set_ylabel('Average Degree (Original Graph)')
    ax.set_title('Avg Degree of Perturbed Nodes')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30)
    ax.legend()

    # 网格线（可选）
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig("avg_degree_perturbed_nodes.pdf")
    plt.show()