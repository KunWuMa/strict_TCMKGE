# -*- coding: utf-8 -*-
"""Compile frozen experiment JSON files into a reviewer-facing Chinese report."""

from __future__ import annotations

import json
from pathlib import Path

from compositional_kge import HERE, OUT


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def pm(metric):
    return f'{metric["mean"]:.4f} ± {metric["sample_std"]:.4f}'


def ci(item):
    bounds = item.get("cluster_bootstrap_95ci", item.get("bootstrap_95ci", item.get("paired_query_bootstrap_95ci")))
    difference = item.get("mrr_difference", item.get("difference"))
    return f'{difference:+.4f} [{bounds[0]:+.4f}, {bounds[1]:+.4f}]'


def main():
    confirm = read("hybrid_confirmatory_results.json")
    detail = read("hybrid_confirmatory_detailed_analysis.json")
    robust = read("expert_systems_robustness.json")
    perm = read("composition_permutation_extended.json")
    transfer = read("decoder_transfer_results.json")
    transfer_stats = read("decoder_transfer_analysis.json")
    dimensions = read("dimension_sensitivity_results.json")
    efficiency = read("efficiency_results.json")
    attribution = read("pair_attribution_cases.json")
    ablation = read("pair_branch_ablation.json")
    external = read("external_zero_shot_results.json")

    methods = [
        ("Hybrid PairComp-KGE", "hybrid_summary"),
        ("等参数双均值对照", "parameter_control_summary"),
        ("DeepSets", "deepsets_summary"),
        ("Pair-only", "pair_only_summary"),
    ]
    lines = [
        "# 面向 Expert Systems 的完整实验结果",
        "",
        "## 可投稿性判断",
        "",
        "当前证据支持一篇以**药对交互感知的组成生成式归纳知识图谱嵌入**为核心的算法论文。最稳健的主张是：在 TCM-MKG 中将未见方剂完全从训练图移除、只能依据组成药物生成表示时，显式二阶药对矩相对结构和参数量完全相同的伪药对对照，显著改善方剂—中医术语链接预测的 MRR 和 Hits@1。方法收益集中于头部排序和小方剂，并在 DistMult 上复现；TransE 和独立门诊处方零样本迁移均未显示显著的药对特异增益。",
        "",
        "独立门诊数据的可行性外部验证已完成。冻结模型明显优于均匀随机期望，但 Hybrid 与等参数对照的 MRR 基本相同，且频率基线略高；因此外部实验只能证明跨资源零样本排序具有可行性，不能证明药对分支在成药到个体处方的强域偏移下仍保留优势。",
        "",
        "## 数据与协议",
        "",
        f'- 确证测试集：{detail["integrity"]["queries"]:,} 条查询，{detail["integrity"]["test_heads"]:,} 个从训练图完全移除的方剂。',
        "- 训练时移除所有开发/测试方剂关联边；测试方剂表示仅由 D4 方剂组成生成。",
        "- 三个固定随机种子；所有模型先按开发集选择检查点，再统一打开测试集。",
        "- 主指标为关系类型约束、filtered MRR；同时报告 Hits@1/3/10。",
        "- 主统计以方剂为聚类单位做 20,000 次 bootstrap；查询级配对 bootstrap 作为补充。",
        "",
        "## 主结果",
        "",
        "| 方法 | MRR | Hits@1 | Hits@3 | Hits@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in methods:
        row = confirm[key]
        lines.append(f'| {label} | {pm(row["mrr"])} | {pm(row["hits1"])} | {pm(row["hits3"])} | {pm(row["hits10"])} |')
    clustered = robust["clustered_bootstrap"]
    lines += [
        "",
        "以 MRR 为主指标的方剂聚类 bootstrap：",
        "",
        f'- Hybrid 对等参数对照：{ci(clustered["hybrid_vs_parameter_control"])}。',
        f'- Hybrid 对 DeepSets：{ci(clustered["hybrid_vs_deepsets"])}；区间轻微跨零。',
        f'- Hybrid 对 Pair-only：{ci(clustered["hybrid_vs_pair_only"])}；二者总体 MRR 无显著差异。',
        "",
        f'完整模型相对等参数对照的 MRR 提高 {100*(confirm["hybrid_summary"]["mrr"]["mean"]-confirm["parameter_control_summary"]["mrr"]["mean"])/confirm["parameter_control_summary"]["mrr"]["mean"]:.2f}%，Hits@1 提高 {confirm["hybrid_summary"]["hits1"]["mean"]-confirm["parameter_control_summary"]["hits1"]["mean"]:+.4f}。Hits@10 反而降低 {confirm["hybrid_summary"]["hits10"]["mean"]-confirm["parameter_control_summary"]["hits10"]["mean"]:+.4f}，说明药对分支主要把正确答案推到榜首，而不是普遍改善长尾召回。',
        "",
        "## 替代解释排除",
        "",
        f'- 错误组成置换：正确组成 MRR={perm["observed_mrr"]:.4f}；50 次保持方剂规模的错误组成零分布为 {perm["null_mrr_mean"]:.4f} ± {perm["null_mrr_std"]:.4f}，单侧经验 p={perm["empirical_p_upper"]:.4f}。',
        f'- 推理时关闭药对分支：MRR 从 {ablation["full"]["mrr"]:.4f} 降至 {ablation["pair_branch_zeroed"]["mrr"]:.4f}，差值 {ablation["full_minus_zeroed"]["mrr"]["difference"]:+.4f}。该结果反映联合训练后的分支依赖，不能单独解释为因果贡献。',
        "- 精确参数等量对照保留相同的双分支、门控和参数量，只把药对矩替换为均值；因此主差异不能用容量增加解释。",
        "",
        "## 缺药稳健性",
        "",
        "| 随机移除组成 | MRR（五掩码均值 ± 掩码间 SD） |",
        "|---:|---:|",
    ]
    for rate, value in robust["ingredient_dropout"].items():
        m = value["mrr"]
        lines.append(f'| {float(rate):.0%} | {m["mean"]:.4f} ± {m["std_across_masks"]:.4f} |')
    lines += [
        "",
        "## 解码器迁移",
        "",
        "| 解码器 | 均值组成 MRR | 均值+药对 MRR | 差值及 95% CI（按方剂聚类） |",
        "|---|---:|---:|---:|",
    ]
    for decoder, label in (("distmult", "DistMult"), ("transe", "TransE")):
        a = transfer[decoder]["mean"]["summary"]["mrr"]
        b = transfer[decoder]["pair"]["summary"]["mrr"]
        stat = transfer_stats["decoders"][decoder]["pair_vs_mean_formula_cluster_bootstrap"]
        lines.append(f'| {label} | {pm(a)} | {pm(b)} | {ci(stat)} |')
    lines += [
        "",
        "DistMult 上收益显著；TransE 上置信区间跨零。因此论文应把结果表述为二阶组成信息与双线性解码器具有稳定互补性，而不是宣称与任意评分函数无关。",
        "",
        "## 独立门诊处方零样本外部验证",
        "",
        f'- 数据：Zenodo `10.5281/zenodo.21951692` v1.1，共 {external["coverage"]["total_prescriptions"]} 张处方。严格术语映射后纳入 {external["coverage"]["included_prescriptions"]} 张处方、{external["coverage"]["included_patients"]} 位患者、{external["coverage"]["evaluation_queries"]} 个证候查询。三个 TCM-MKG 检查点完全冻结，未用外部标签训练或调参。',
        f'- 映射覆盖：中药类型 {external["coverage"]["herb_unique_exact_coverage"]:.1%}，中药记录 {external["coverage"]["herb_record_exact_coverage"]:.1%}，已记录证候字符串 {external["coverage"]["syndrome_unique_exact_coverage"]:.1%}。因有效处方不足 100 且证候类型覆盖不足 50%，按预设规则定性为可行性验证。',
        "",
        "| 方法 | MRR | Hits@1 | Hits@3 | Hits@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    external_methods = [
        ("Hybrid PairComp-KGE", "hybrid"),
        ("等参数双均值对照", "parameter_control"),
        ("DeepSets", "deepsets"),
        ("Pair-only", "pair_only"),
    ]
    for label, key in external_methods:
        row = external["overall"][key]
        lines.append(f'| {label} | {pm(row["mrr"])} | {pm(row["hits1"])} | {pm(row["hits3"])} | {pm(row["hits10"])} |')
    lines.append(f'| 训练队列频率基线 | {external["frequency"]["mrr"]:.4f} | {external["frequency"]["hits1"]:.4f} | {external["frequency"]["hits3"]:.4f} | {external["frequency"]["hits10"]:.4f} |')
    lines.append(f'| 均匀随机期望 | {external["uniform_random_expectation"]["metrics"]["mrr"]:.4f} | {external["uniform_random_expectation"]["metrics"]["hits1"]:.4f} | {external["uniform_random_expectation"]["metrics"]["hits3"]:.4f} | {external["uniform_random_expectation"]["metrics"]["hits10"]:.4f} |')
    ext_pc = external["patient_cluster_bootstrap"]["hybrid_vs_parameter_control"]
    ext_ds = external["patient_cluster_bootstrap"]["hybrid_vs_deepsets"]
    lines += [
        "",
        f'- Hybrid 对等参数对照：MRR 差 {ext_pc["mrr"]["difference"]:+.4f}，患者聚类 95% CI [{ext_pc["mrr"]["patient_cluster_bootstrap_95ci"][0]:+.4f}, {ext_pc["mrr"]["patient_cluster_bootstrap_95ci"][1]:+.4f}]；Hits@1 差 {ext_pc["hits1"]["difference"]:+.4f}，区间同样跨零。',
        f'- Hybrid 对 DeepSets：MRR 差 {ext_ds["mrr"]["difference"]:+.4f}，患者聚类 95% CI [{ext_ds["mrr"]["patient_cluster_bootstrap_95ci"][0]:+.4f}, {ext_ds["mrr"]["patient_cluster_bootstrap_95ci"][1]:+.4f}]。',
        "- 结论：模型保留了高于随机的排序信号，但没有超过简单频率基线，也没有在外部样本上确认药对特异增益。该结果应作为从中成药知识图谱迁移到个体化处方的域偏移边界，而不是算法有效性的外部阳性证据。",
        "",
        "## 嵌入维度敏感性",
        "",
        "| 维度 | MRR | Hits@1 | 参数量 |",
        "|---:|---:|---:|---:|",
    ]
    for dim in (64, 128, 256):
        value = dimensions["dimensions"][str(dim)]
        params = (f'{efficiency["models"]["hybrid"]["parameters"]:,}' if dim == 128
                  else f'{next(iter(value["seeds"].values()))["parameters"]:,}')
        lines.append(f'| {dim} | {pm(value["summary"]["mrr"])} | {pm(value["summary"]["hits1"])} | {params} |')
    overhead = efficiency["hybrid_over_deepsets"]
    lines += [
        "",
        "## 效率",
        "",
        "| 编码器 | 参数量 | 610 个未见方剂批量编码延迟（ms，三种子均值 ± SD） |",
        "|---|---:|---:|",
    ]
    display = {"hybrid": "Hybrid PairComp", "parameter_control": "等参数对照", "deepsets": "DeepSets", "pair_only": "Pair-only"}
    for key, label in display.items():
        value = efficiency["models"][key]
        lines.append(f'| {label} | {value["parameters"]:,} | {value["batch_milliseconds_mean"]:.4f} ± {value["batch_milliseconds_std"]:.4f} |')
    lines += [
        "",
        f'Hybrid 相对 DeepSets 只增加 {overhead["additional_parameters"]:,} 个参数（{overhead["relative_parameter_increase_percent"]:.3f}%）。利用平方和恒等式，全部药对矩由显式 $O(n^2d)$ 枚举降为 $O(nd)$。',
        "",
        "## 解释案例",
        "",
        f'共找到 {attribution["eligible_edges"]} 条满足方剂含 2–10 味药且三个种子均预测为第 1 名的候选边，从中选择药对分支正向干预最大的五个不同方剂。详细药名、术语及正负药对归因位于 `results/pair_attribution_cases.json`。这些归因是模型忠实度解释，不作临床配伍有效性结论。',
        "",
        "## 论文应使用的贡献表述",
        "",
        "1. 提出一个面向未见复方实体的组成生成式归纳 KGE，在复数嵌入空间同时编码一阶药物语义和二阶药对矩。",
        "2. 用平方和恒等式在线性时间内聚合全部有序药对，并用方剂规模门控控制高基数二阶矩。",
        "3. 建立严格的方剂实体留出协议，并以精确参数等量对照、错误组成置换和方剂聚类统计识别药对输入的真实增益。",
        "4. 揭示收益边界：小方剂和双线性解码器收益最稳定，Hits@10 与 TransE 并未显著改善。",
        "5. 在独立门诊处方上执行预先规定的冻结零样本压力测试，报告未获阳性的药对特异迁移结果及术语覆盖限制。",
        "",
        "## 投稿前剩余硬任务",
        "",
        "- 将代码、冻结划分、环境文件和结果清单上传到公开仓库并归档 DOI/URL，以满足期刊数据共享要求。",
        "- 由中医专业人员对五个案例的药对解释做盲评可增强临床解释，但不是算法主结论成立的必要条件；没有专家时必须限定为模型内解释。",
        "",
        "## Expert Systems 格式检查",
        "",
        "- 原创论文总字数不超过 15,000；摘要不超过 250 词，并明确相对现有工作的增量。",
        "- 4–7 个关键词；正文含数据可用性、资金、利益冲突和适用时的伦理声明。",
        "- 表中明确 SD/SEM；图表均有可独立理解的图注；初投可自由格式，建议 LaTeX。",
        "- 投稿说明：https://onlinelibrary.wiley.com/page/journal/14680394/homepage/forauthors.html",
    ]
    target = HERE / "EXPERT_SYSTEMS_COMPLETE_RESULTS.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
