from scipy.stats import expon, uniform
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

#################################### Intensity & Simulation ####################################

def intensity(tau, beta, c):
    """
    条件强度在“距上次事件的时间” tau 上的函数（age-based intensity）：
        λ(τ) = c / (1 + beta * exp(c * τ)),   τ >= 0

    在点过程层面：
        给定历史 H_{n-1}，若上一次事件时间为 T_{n-1}（且 n>=2），
        则在 t > T_{n-1} 时的条件强度为
            λ_n(t | H_{n-1}) = λ(τ)  with  τ = t - T_{n-1}.

    注意：第一个点（n=1）我们单独用常数强度 c0 处理，不走这个函数。
    """
    beta = float(beta)
    c = float(c)
    tau = np.asarray(tau, dtype=float)
    return c / (1.0 + beta * np.exp(c * tau))


def multi_hawkes_simulation(beta, c, T, c0=None):
    """
    模拟一维“历史依赖”的点过程：

    - 第一个点：强度是常数 c0，对应 inter-arrival τ_1 ~ Exp(c0)；
    - 后续点（n >= 2）：条件强度依赖于距上次事件的时间 τ：
          λ_n(t | H_{n-1}) = c / (1 + beta * exp(c * (t - T_{n-1})))

      这对应的 inter-arrival 分布为：
        - 以概率 p_finite = 1 / (1 + beta) 产生一个有限的等待时间，
          条件下该有限等待时间 ~ Exp(c)；
        - 以概率 1 - p_finite = beta / (1 + beta) 再也没有下一次事件（该 inter-arrival = +∞）。

    我们利用这个显式分布直接采样：
        - 第一段：纯 Exp(c0)；
        - 之后每段：先用 Bernoulli 判断是否 finite，再用 Exp(c) 给出长度。

    输入:
        beta: 标量 > 0
        c:    标量 > 0（logistic 部分的 scale）
        T:    截止时间
        c0:   第一个点的常数强度（标量 > 0），若为 None 则默认 c0 = c

    输出:
        points_hawkes: 长度为 1 的 list，
                       points_hawkes[0] 是所有事件时间的升序列表
    """
    beta = float(beta)
    c = float(c)
    T = float(T)
    if c0 is None:
        c0 = c
    c0 = float(c0)

    # logistic 部分的 finite 概率
    p_finite = 1.0 / (1.0 + beta)

    t = 0.0
    points = []

    # ---------- 第一个点：强度 c0，等待时间 ~ Exp(c0) ----------
    r1 = expon.rvs(scale=1.0 / c0)
    t = t + r1
    if t > T:
        # 整个窗口里一个点都没有
        return [points]

    points.append(t)

    # ---------- 后续点：conditional-logistic 型 renewal ----------
    while t < T:
        # 先决定本段 inter-arrival 是否 finite
        u = uniform.rvs(loc=0.0, scale=1.0)
        if u >= p_finite:
            # 这个间隔是 +∞，后面再也没有点
            break

        # finite 的话，等待时间 ~ Exp(c)
        r = expon.rvs(scale=1.0 / c)
        t = t + r
        if t > T:
            break

        points.append(t)

    # 保持原返回格式：list of list
    points_hawkes = [points]
    return points_hawkes


#################################### Simulation ####################################

n_realizations = 2000
T = 50.0

# 现在只有一个类别
M = 1

beta = 0.01
c = 2.0

# 第一个点的常数强度 c0（你可以单独调这个来控制 0 序列概率）
c0 = 2.0   # 例如取和 c 一样；取大一点会让 0 序列几乎消失

print("beta =", beta, "c =", c, "c0 =", c0)

time_seqs = []
type_seqs = []
num_empty = 0  # 统计空序列数量

for seq_idx in range(n_realizations):
    print(f'{seq_idx}-th seqs')

    points_hawkes = multi_hawkes_simulation(beta, c, T, c0=c0)  # [points]

    # 一维，所以只看 points_hawkes[0]
    times = np.array(points_hawkes[0], dtype=float)

    # 如果这个 realization 里没有事件，直接丢掉
    if times.size == 0:
        num_empty += 1
        continue

    times = times.reshape(-1, 1)
    # 只有一个类型，就全是 0
    types = np.zeros_like(times, dtype=int)

    concatenation = np.concatenate((times, types), axis=-1)

    # 按时间排序（理论上已经升序，但保险起见）
    concatenation = concatenation[concatenation[:, 0].argsort()]

    time_seqs.append(concatenation[:, 0])
    type_seqs.append(concatenation[:, 1])

print("======================================")
print(f"Total simulated sequences (target): {n_realizations}")
print(f"Sequences kept (with >=1 event):   {len(time_seqs)}")
print(f"Empty sequences removed:           {num_empty}")
print("Empirical zero-sequence ratio:     {num_empty / n_realizations:.6f}")
print("======================================")

#################################### Split, Format, Save ####################################

# 用实际保留下来的数量来做切分
N_total = len(time_seqs)
if N_total == 0:
    raise RuntimeError("所有序列都是空的，没法构造数据集，请检查参数 beta, c, c0, T。")

tr_len = int(N_total * 0.5)
dev_len = int(N_total * 0.25)
test_len = N_total - tr_len - dev_len  # 保证总数对得上

tr_times = time_seqs[:tr_len]
dev_times = time_seqs[tr_len:tr_len + dev_len]
test_times = time_seqs[tr_len + dev_len:]

tr_types = type_seqs[:tr_len]
dev_types = type_seqs[tr_len:tr_len + dev_len]
test_types = type_seqs[tr_len + dev_len:]


def format_converter(times, types):
    batch_ = len(times)
    results = []

    for seq_idx in range(batch_):
        seq_time = np.array(times[seq_idx], dtype=float)
        seq_type = np.array(types[seq_idx], dtype=int)

        # 假设已经去掉空序列，所以这里 len(seq_time) >= 1
        seq_diff = [seq_time[0]] + list(seq_time[1:] - seq_time[:-1])

        result_seq = []
        for i in range(len(seq_time)):
            event = {
                'time_since_start': float(seq_time[i]),
                'time_since_last_event': float(seq_diff[i]),
                'type_event': int(seq_type[i])
            }
            result_seq.append(event)
        results.append(result_seq)
    return results


tr_results = format_converter(tr_times, tr_types)
dev_results = format_converter(dev_times, dev_types)
test_results = format_converter(test_times, test_types)

_to_dump_train_ = {'train': tr_results, 'test': [], 'dev': [], 'dim_process': M}
_to_dump_test_ = {'train': [], 'test': test_results, 'dev': [], 'dim_process': M}
_to_dump_dev_ = {'train': [], 'test': [], 'dev': dev_results, 'dim_process': M}

with open('train.pkl', 'wb') as f:
    pkl.dump(_to_dump_train_, f)

with open('test.pkl', 'wb') as f:
    pkl.dump(_to_dump_test_, f)

with open('dev.pkl', 'wb') as f:
    pkl.dump(_to_dump_dev_, f)


#################################### Visualization ####################################

visualize = True
if visualize and len(time_seqs) > 0:
    # 就画第一条保留下来的序列
    example_times = list(time_seqs[0])
    points_hawkes_plot = [example_times]

    # 事件点位置图
    plt.figure(figsize=(7, 5))
    plt.subplot(1, 1, 1)
    plt.plot([0, T], [0, 0], 'r-', lw=1, alpha=0.6)
    plt.plot(points_hawkes_plot[0],
             [0] * len(points_hawkes_plot[0]),
             linestyle='None',
             marker='|',
             markersize=10,
             label='process 1')
    plt.title('1D renewal process with first-constant then logistic intensity')
    plt.ylim(-1, 1)
    plt.legend(loc='best', frameon=False)
    plt.savefig('hawkes_points.png')

    # 沿着这条样本路径画出条件强度 λ(t | H_{t-})
    x = np.linspace(0, T, 2000)
    lam = np.zeros_like(x)

    # 用 example_times 恢复历史
    first_event_time = example_times[0]
    last_event_time = 0.0
    idx_event = 0
    num_events = len(example_times)

    for i, t in enumerate(x):
        if t < first_event_time:
            # 第一个事件之前：强度是常数 c0
            lam[i] = c0
            continue

        # 更新“当前 t 之前的最后一次事件时间”
        while idx_event < num_events and example_times[idx_event] <= t:
            last_event_time = example_times[idx_event]
            idx_event += 1

        tau = t - last_event_time  # age
        lam[i] = intensity(tau, beta, c)

    plt.figure(figsize=(7, 5))
    plt.plot(x, lam, 'r-', label='conditional intensity λ(t | H_{t-})')
    plt.ylim(0)
    plt.legend(loc='best', frameon=False)
    plt.title('First-constant then conditional-logistic intensity')
    plt.savefig('multi-hawkes_intensity.png')
    plt.close()


# from scipy.stats import expon, uniform
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle as pkl

# #################################### Intensity & Simulation ####################################

# def intensity(tau, beta, c):
#     """
#     条件强度在“距上次事件的时间” tau 上的函数（age-based intensity）：
#         λ(τ) = c / (1 + beta * exp(c * τ)),   τ >= 0

#     在点过程层面：
#         给定历史 H_{n-1}，若上一次事件时间为 T_{n-1}，
#         则在 t > T_{n-1} 时的条件强度为
#             λ_n(t | H_{n-1}) = λ(τ)  with  τ = t - T_{n-1}.
#     """
#     beta = float(beta)
#     c = float(c)
#     tau = np.asarray(tau, dtype=float)
#     return c / (1.0 + beta * np.exp(c * tau))


# def multi_hawkes_simulation(beta, c, T):
#     """
#     模拟一维“历史依赖”的点过程，条件强度依赖于距上次事件的时间 τ：

#         λ_n(t | H_{n-1}) = c / (1 + beta * exp(c * (t - T_{n-1})))

#     这个过程实际上是一个 renewal 过程，其 inter-arrival 时间分布为：
#         - 以概率 p_finite = 1 / (1 + beta) 产生一个有限的等待时间，
#           条件下该有限等待时间 ~ Exp(c)；
#         - 以概率 1 - p_finite = beta / (1 + beta) 再也没有下一次事件（间隔为 +∞）。

#     我们利用这个显式分布直接采样，避免再做 thinning。

#     输入:
#         beta: 标量 > 0
#         c:    标量 > 0
#         T:    截止时间

#     输出:
#         points_hawkes: 长度为 1 的 list，
#                        points_hawkes[0] 是所有事件时间的升序列表
#     """
#     beta = float(beta)
#     c = float(c)

#     # 每个 inter-arrival 有概率 1/(1+beta) 是“有限”，概率 beta/(1+beta) 是“无限”
#     p_finite = 1.0 / (1.0 + beta)

#     t = 0.0
#     points = []

#     while t < T:
#         # 先决定是否还有下一次 finite 的事件
#         u = uniform.rvs(loc=0.0, scale=1.0)
#         if u >= p_finite:
#             # 没有下一次事件（间隔为 +∞），直接结束
#             break

#         # 有下一次事件，则等待时间 ~ Exp(c)
#         r = expon.rvs(scale=1.0 / c)
#         t = t + r
#         if t > T:
#             break

#         points.append(t)

#     # 保持原返回格式：list of list
#     points_hawkes = [points]
#     return points_hawkes


# #################################### Simulation ####################################

# n_realizations = 4000
# T = 10.0

# # 现在只有一个类别
# M = 1

# beta = 0.01
# c = 2.0

# print("beta =", beta, "c =", c)

# time_seqs = []
# type_seqs = []
# num_empty = 0  # 统计空序列数量

# for seq_idx in range(n_realizations):
#     print(f'{seq_idx}-th seqs')

#     points_hawkes = multi_hawkes_simulation(beta, c, T)  # [points]

#     # 一维，所以只看 points_hawkes[0]
#     times = np.array(points_hawkes[0], dtype=float)

#     # 如果这个 realization 里没有事件，直接丢掉
#     if times.size == 0:
#         num_empty += 1
#         continue

#     times = times.reshape(-1, 1)
#     # 只有一个类型，就全是 0
#     types = np.zeros_like(times, dtype=int)

#     concatenation = np.concatenate((times, types), axis=-1)

#     # 按时间排序（理论上已经升序，但保险起见）
#     concatenation = concatenation[concatenation[:, 0].argsort()]

#     time_seqs.append(concatenation[:, 0])
#     type_seqs.append(concatenation[:, 1])

# print("======================================")
# print(f"Total simulated sequences (target): {n_realizations}")
# print(f"Sequences kept (with >=1 event):   {len(time_seqs)}")
# print(f"Empty sequences removed:           {num_empty}")
# print("======================================")

# #################################### Split, Format, Save ####################################

# # 用实际保留下来的数量来做切分
# N_total = len(time_seqs)
# if N_total == 0:
#     raise RuntimeError("所有序列都是空的，没法构造数据集，请检查参数 beta, c, T。")

# tr_len = int(N_total * 0.5)
# dev_len = int(N_total * 0.25)
# test_len = N_total - tr_len - dev_len  # 保证总数对得上

# tr_times = time_seqs[:tr_len]
# dev_times = time_seqs[tr_len:tr_len + dev_len]
# test_times = time_seqs[tr_len + dev_len:]

# tr_types = type_seqs[:tr_len]
# dev_types = type_seqs[tr_len:tr_len + dev_len]
# test_types = type_seqs[tr_len + dev_len:]


# def format_converter(times, types):
#     batch_ = len(times)
#     results = []

#     for seq_idx in range(batch_):
#         seq_time = np.array(times[seq_idx], dtype=float)
#         seq_type = np.array(types[seq_idx], dtype=int)

#         # 假设已经去掉空序列，所以这里 len(seq_time) >= 1
#         seq_diff = [seq_time[0]] + list(seq_time[1:] - seq_time[:-1])

#         result_seq = []
#         for i in range(len(seq_time)):
#             event = {
#                 'time_since_start': float(seq_time[i]),
#                 'time_since_last_event': float(seq_diff[i]),
#                 'type_event': int(seq_type[i])
#             }
#             result_seq.append(event)
#         results.append(result_seq)
#     return results


# tr_results = format_converter(tr_times, tr_types)
# dev_results = format_converter(dev_times, dev_types)
# test_results = format_converter(test_times, test_types)

# _to_dump_train_ = {'train': tr_results, 'test': [], 'dev': [], 'dim_process': M}
# _to_dump_test_ = {'train': [], 'test': test_results, 'dev': [], 'dim_process': M}
# _to_dump_dev_ = {'train': [], 'test': [], 'dev': dev_results, 'dim_process': M}

# with open('train.pkl', 'wb') as f:
#     pkl.dump(_to_dump_train_, f)

# with open('test.pkl', 'wb') as f:
#     pkl.dump(_to_dump_test_, f)

# with open('dev.pkl', 'wb') as f:
#     pkl.dump(_to_dump_dev_, f)


# #################################### Visualization ####################################

# visualize = True
# if visualize and len(time_seqs) > 0:
#     # 就画第一条保留下来的序列
#     example_times = list(time_seqs[0])
#     points_hawkes_plot = [example_times]

#     # 事件点位置图
#     plt.figure(figsize=(7, 5))
#     plt.subplot(1, 1, 1)
#     plt.plot([0, T], [0, 0], 'r-', lw=1, alpha=0.6)
#     plt.plot(points_hawkes_plot[0],
#              [0] * len(points_hawkes_plot[0]),
#              linestyle='None',
#              marker='|',
#              markersize=10,
#              label='process 1')
#     plt.title('1D renewal process with history-dependent intensity')
#     plt.ylim(-1, 1)
#     plt.legend(loc='best', frameon=False)
#     plt.savefig('hawkes_points.png')

#     # 沿着这条样本路径画出条件强度 λ(t | H_{t-})
#     x = np.linspace(0, T, 2000)
#     lam = np.zeros_like(x)

#     last_event_time = 0.0
#     idx_event = 0
#     num_events = len(example_times)

#     for i, t in enumerate(x):
#         # 更新“当前 t 之前的最后一次事件时间”
#         while idx_event < num_events and example_times[idx_event] <= t:
#             last_event_time = example_times[idx_event]
#             idx_event += 1

#         tau = t - last_event_time  # age
#         lam[i] = intensity(tau, beta, c)

#     plt.figure(figsize=(7, 5))
#     plt.plot(x, lam, 'r-', label='conditional intensity λ(t | H_{t-})')
#     plt.ylim(0)
#     plt.legend(loc='best', frameon=False)
#     plt.title('History-dependent intensity along one realization')
#     plt.savefig('multi-hawkes_intensity.png')
