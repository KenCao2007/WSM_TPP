# 为什么 Survival-Augmented TLL 会变正？

对应论文：Score Matching for Estimating Finite Point Processes (arXiv 2512.04617)

---

## 1. 标准 MLE Log-likelihood（时间维度）

对于观测序列 $(t_1, \ldots, t_N)$，标准 MLE log-likelihood 为（论文 Eq. 3 的时间部分）：

$$ll_{\text{MLE}} = \sum_{n=1}^{N} \log \lambda_n(t_n \mid \mathcal{H}_{n-1}) - \sum_{n=1}^{N+1}\int_{t_{n-1}}^{t_n} \lambda_n(t \mid \mathcal{H}_{n-1})\, dt$$

其中 $t_0 = 0$，$t_{N+1} = T$。记 $\Lambda_n = \int_{t_{n-1}}^{t_n}\lambda_n\,dt$，则可以简写为：

$$ll_{\text{MLE}} = \sum_{n=1}^{N} \log \lambda_n(t_n \mid \mathcal{H}_{n-1}) - \int_0^T \lambda(t)\, dt$$

**关键性质**：由 $\mathrm{KL}(p \| q) \geq 0$，ground truth 参数在期望意义下 maximize $ll_{\text{MLE}}$，且该值**永远为负**。

---

## 2. Survival-Augmented Log-likelihood（论文 Eq. 18）

加入 survival classification 后，log-likelihood 变为：

$$ll_T = \sum_{n=1}^{N} \Big[\underbrace{\log \tilde{\lambda}_n - \tilde{\Lambda}_n - \log\!\left(1 - \tilde{G}_n(T)\right) + \log \hat{F}_n}_{\text{likelihood of } p_n(t_n \mid \mathcal{H}_{n-1})}\Big] + \underbrace{\log\!\left(1 - \hat{F}_{N+1}\right)}_{\text{likelihood of } p(N(\mathcal{X})=N \mid \mathcal{H}_N)}$$

其中各符号定义为：

$$\tilde{\Lambda}_n = \int_{t_{n-1}}^{t_n} \tilde{\lambda}_{T,n}(\tau \mid \mathcal{H}_{n-1})\, d\tau \qquad \text{（从上一事件到当前事件的 compensator）}$$

$$\tilde{\Lambda}_n(T) = \int_{t_{n-1}}^{T} \tilde{\lambda}_{T,n}(\tau \mid \mathcal{H}_{n-1})\, d\tau \qquad \text{（从上一事件到窗口末端的 compensator）}$$

$$\tilde{G}_n(T) = \exp\!\left(-\tilde{\Lambda}_n(T)\right) \qquad \text{（surrogate intensity 诱导的 survival probability）}$$

$$\hat{F}_n(\mathcal{H}_{n-1}) = \Pr(\hat{N} \geq n \mid \mathcal{H}_{n-1}) \in [0,1] \qquad \text{（learned：序列至少还有第 } n \text{ 个事件的概率）}$$

---

## 3. 为什么 GT 参数下两个公式等价？

在 ground truth 参数 $\theta^*$ 下，$\tilde{\lambda} = \lambda^*$，此时 survival classifier 收敛到真实值：

$$\hat{F}_n(\mathcal{H}_{n-1}) \to \Pr(N \geq n \mid \mathcal{H}_{n-1}) = 1 - G_n^*(T \mid \mathcal{H}_{n-1})$$

即 $\hat{F}_n = 1 - \tilde{G}_n(T)$。代入 $ll_T$ 中的每个事件 $n$：

$$-\log\!\left(1 - \tilde{G}_n(T)\right) + \log \hat{F}_n = -\log\!\left(1 - \tilde{G}_n(T)\right) + \log\!\left(1 - \tilde{G}_n(T)\right) = 0$$

**这两项精确相消。**

最后一项：

$$\log\!\left(1 - \hat{F}_{N+1}\right) = \log G_{N+1}^*(T \mid \mathcal{H}_N) = -\tilde{\Lambda}_{N+1}(T)$$

所以 $ll_T$ 退化为：

$$ll_T = \sum_{n=1}^{N} \log \lambda_n^*(t_n \mid \mathcal{H}_{n-1}) - \big[\tilde{\Lambda}_1 + \tilde{\Lambda}_2 + \cdots + \tilde{\Lambda}_N + \tilde{\Lambda}_{N+1}(T)\big] = ll_{\text{MLE}}$$

**结论：Survival-augmented 公式在 GT 处与标准 MLE 完全等价，mathematically consistent。**

---

## 4. 训练发散时为什么 TLL 变正？

### WSM Objective 无下界

WSM/AWSM 的 temporal 部分（论文 Eq. 14）为：

$$\mathcal{J}^{\text{AWSM,T}}_{h_T}(\theta) = \mathbb{E}\left\{\sum_{n=1}^{N(\mathcal{X})} \left[\frac{1}{2}\psi_{T,n,\theta}^2\, h_T + \partial_{t_n}\psi_{T,n,\theta}\, h_T + \psi_{T,n,\theta}\,\partial_{t_n}h_T\right]\right\}$$

其中 conditional score function（论文 Eq. 10）为：

$$\psi_{T,n}(t_n \mid \mathcal{H}_{n-1}) = \partial_{t_n}\log\lambda_{T,n}(t_n \mid \mathcal{H}_{n-1}) - \lambda_{T,n}(t_n \mid \mathcal{H}_{n-1})$$

$\mathcal{J}^{\text{AWSM,T}}$ **在理论上 unbounded below**，可以沿某些方向趋向 $-\infty$，导致 intensity 崩溃到接近 0。

### 逐项分析（令 $\lambda \to \varepsilon \to 0$）

设训练发散后 intensity 退化为常数 $\varepsilon$（极小），对每个事件 $n$：

$$\text{(1)}\quad \log \tilde{\lambda}_n = \log \varepsilon \qquad \to -\infty$$

$$\text{(2)}\quad \tilde{\Lambda}_n = \varepsilon \cdot (t_n - t_{n-1}) \qquad \to 0$$

$$\text{(3)}\quad \tilde{\Lambda}_n(T) = \varepsilon \cdot (T - t_{n-1}) \qquad \to 0$$

$$\tilde{G}_n(T) = e^{-\varepsilon(T-t_{n-1})} \approx 1 - \varepsilon(T-t_{n-1})$$

$$1 - \tilde{G}_n(T) \approx \varepsilon\,(T - t_{n-1})$$

$$-\log\!\left(1 - \tilde{G}_n(T)\right) \approx -\log\varepsilon - \log(T-t_{n-1}) \qquad \to +\infty$$

将三项相加，$\log\varepsilon$ 和 $-\log\varepsilon$ **相消**：

$$\log\varepsilon \;+\; 0 \;+\; \big(-\log\varepsilon - \log(T-t_{n-1})\big) \;=\; -\log(T - t_{n-1})$$

**这个极限值与 $\varepsilon$ 无关**，只取决于从上一事件到窗口末端的剩余时间 $T - t_{n-1}$。

### 何时为正？

$$-\log(T - t_{n-1}) > 0 \quad \Longleftrightarrow \quad T - t_{n-1} < 1$$

Retweet 数据集使用 `normalize_scale = 50`，归一化后大量事件的剩余时间小于 1，所以对所有事件求和后 per-event TLL 变正。

---

## 5. 直觉解释

$-\log(1-\tilde{G}_n(T))$ 是一个**条件化归一化项**：

- 模型预测 $\lambda \approx 0$，"几乎不可能有事件发生"
- 但 test data 里事件确实发生了
- 给定事件发生（$N \geq n$），在 $(t_{n-1}, T)$ 内的条件密度为

$$p_n(t_n \mid \mathcal{H}_{n-1}, N \geq n) = \frac{\tilde{\lambda}_n(t_n)\,e^{-\tilde{\Lambda}_n(t_n)}}{1 - \tilde{G}_n(T)}$$

当 $\lambda \to 0$ 时，这退化为 $(T - t_{n-1})^{-1}$（在剩余时间窗口内的均匀分布）。

当 $T - t_{n-1} < 1$ 时，密度 $> 1$，$\log > 0$。

**这不代表模型好**——只是训练不稳定的数值副作用，而非 formula bug。

---

## 6. 总结

| 情况 | $ll_T$ 特征 |
|------|------------|
| 模型收敛到 $\theta^*$ | $ll_T = ll_{\text{MLE}} < 0$，额外项精确相消 |
| 模型部分收敛 | $ll_T < 0$，轻微偏差 |
| **模型发散（$\lambda \to 0$）** | **$ll_T > 0$，$-\log(1-\tilde{G}_n(T))$ 主导** |

**因果链**：WSM unbounded below $\to$ 训练不稳定 $\to$ $\lambda \to 0$ $\to$ $\hat{F}_n$ 与 $1-\tilde{G}_n(T)$ 不再相等 $\to$ 两项未相消 $\to$ $ll_T > 0$

---

## 7. 解决方案

```python
# 在 optimizer.step() 之前加入 gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

| 方案 | 说明 |
|------|------|
| Gradient clipping | 防止单步梯度过大推入 degenerate 区域 |
| Early stopping | val TLL 变正或跌幅异常时停训，取最优 checkpoint |
| 增大 `alpha_survival` | 从 10 调至 50~100；survival loss 有界，给训练加稳定锚 |
| 多 seed 实验 | 跑 5 个 seed，报 mean ± std，标注发散的 run |
