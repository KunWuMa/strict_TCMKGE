# 药对交互感知组成生成式归纳KGE：算法与实验协议

## 1. 假设

等权组成编码器只保留饮片嵌入的一阶均值。两个方剂即使药物配伍不同，也可能得到相近均值；中医方剂语义又经常取决于药物之间的共同作用。因此，引入二阶药对交互表示，检验其是否能在完全未见方剂的诊断关系补全中稳定超过一阶平均。

## 2. PairComp-KGE

设饮片复嵌入为`z_h = a_h + i b_h`。一阶表示为`mu_f = mean(z_h)`。二阶药对表示定义为所有不同饮片有序对的复Hadamard乘积均值：

`p_f = 1/(n(n-1)) * sum_{i != j} (z_i elementwise-multiply z_j)`。

利用恒等式可在`O(nd)`而非`O(n^2 d)`时间内计算：

- `Re(p) = ((sum a)^2 - sum(a^2) - (sum b)^2 + sum(b^2)) / (n(n-1))`
- `Im(p) = 2*((sum a)(sum b) - sum(ab)) / (n(n-1))`

通过可学习的复线性变换`W`和残差门控融合：

`z_f = mu_f + sigmoid(g_r) * tanh(W p_f)`。

`g_r`按查询关系学习，使同一药对对疾病、证候和治法可以有不同贡献。单味方剂的二阶项定义为零。模型仍使用ComplEx评分函数，测试方剂不使用ID嵌入。

## 3. 开发阶段固定比较

1. `uniform_mean`：一阶均值强基线；
2. `deepsets_residual`：对一阶均值增加参数量相近的复MLP残差，控制收益是否仅来自更多参数；
3. `pair_residual`：二阶药对表示，使用全局门控；
4. `relation_pair_residual`：二阶药对表示，使用关系门控，即主候选。

开发阶段只使用已打开过测试集的`composition_holdout`中的训练和开发部分，绝不再次依据其测试指标选择结构。最大40轮，开发MRR选checkpoint，其他超参数沿用原实验。

## 4. 新外层盲测

开发阶段选定结构后，使用新种子构建`pair_holdout`，仅从具有D3目标边和D4组成的方剂中按方剂互斥划分。至少运行：选定PairComp-KGE三种子、uniform_mean三种子、参数匹配DeepSets三种子、频率、严格ID ComplEx和上下文ComplEx上界。所有checkpoint冻结后统一打开测试。

只有当PairComp-KGE在新盲测上相对`uniform_mean`的配对bootstrap区间不跨0，且三种子均值更高，才将“药对交互提高性能”写为主要算法贡献。否则保留为未获支持的假设。
