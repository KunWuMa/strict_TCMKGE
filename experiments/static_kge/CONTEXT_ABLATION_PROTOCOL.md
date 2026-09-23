# 方剂冷启动上下文消融协议

## 目的

判断方剂留出测试中的KGE收益来自哪一类训练上下文。模型固定为首轮最优的ComplEx，嵌入维度、学习率、负样本数、开发选择间隔和最大轮数与首轮完全一致。

## 预先固定的路线

| 路线 | 移除的训练关系 | 检验含义 |
|---|---|---|
| full | 无 | 首轮完整图参照，不重复训练 |
| no_formula_herb | `cpm_contains_chp` | 去除方剂组成入口，保留方剂—ICD |
| no_icd | `cpm_treats_icd11` | 去除现代疾病入口，保留方剂组成 |
| herb_identity_only | `cpm_treats_icd11`及三类饮片下游关系 | 只保留方剂—饮片身份，不使用药性、物种和化合物 |
| no_formula_context | `cpm_contains_chp`和`cpm_treats_icd11` | 留出方剂没有任何训练边，作为无方剂信息下界 |

目标中成药—中医术语训练边在所有路线相同。开发/测试目标边不变。全部路线先按开发集选择checkpoint，选择全部冻结后才计算测试。

## 主指标与判断门槛

主指标是方剂留出、类型限定、filtered MRR；同时报告Hits@1/3/10和全图候选MRR。

组成感知方向只有在以下条件同时满足时才进入新模型研发：

1. `full`或`no_icd`明显高于`no_formula_herb`与`no_formula_context`；
2. `no_icd`不低于频率基线，说明饮片组成自身包含诊疗信号；
3. `herb_identity_only`与`no_icd`的差异能够量化饮片下游属性是否有增量。

本轮单种子只作机制筛查。若门槛通过，再对选定对照和新模型运行三个训练种子。
