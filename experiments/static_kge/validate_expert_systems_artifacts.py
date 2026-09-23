"""Fail-fast consistency checks for the complete Expert Systems experiment package."""

from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main():
    confirm = read("hybrid_confirmatory_results.json")
    coverage = read("candidate_coverage_audit.json")
    assert coverage["splits"]["dev"]["pre_filter"]["queries"] == 1139
    assert coverage["splits"]["dev"]["excluded"]["queries"] == 25
    assert coverage["splits"]["dev"]["retained"]["queries"] == 1114
    assert coverage["splits"]["test"]["pre_filter"]["queries"] == 1118
    assert coverage["splits"]["test"]["excluded"]["queries"] == 25
    assert coverage["splits"]["test"]["retained"]["queries"] == 1093
    assert coverage["methods_share_identical_retained_queries"]
    assert all(split["builder_filter_equals_relation_specific_coverage"] for split in coverage["splits"].values())
    robust = read("expert_systems_robustness.json")
    permutation = read("composition_permutation_extended.json")
    transfer = read("decoder_transfer_analysis.json")
    dims = read("dimension_sensitivity_results.json")
    attribution = read("pair_attribution_cases.json")
    efficiency = read("efficiency_results.json")
    external = read("external_zero_shot_results.json")
    mapping = read("external_mapping_audit.json")
    assert confirm["hybrid_summary"]["mrr"]["mean"] > confirm["parameter_control_summary"]["mrr"]["mean"]
    primary_ci = robust["clustered_bootstrap"]["hybrid_vs_parameter_control"]["cluster_bootstrap_95ci"]
    assert primary_ci[0] > 0
    assert permutation["permutations"] == 999 and permutation["empirical_p_upper"] <= 0.001
    assert permutation["observed_mrr"] > max(permutation["null_mrr_values"])
    assert transfer["decoders"]["distmult"]["pair_vs_mean_formula_cluster_bootstrap"]["cluster_bootstrap_95ci"][0] > 0
    assert transfer["decoders"]["transe"]["pair_vs_mean_formula_cluster_bootstrap"]["cluster_bootstrap_95ci"][0] < 0
    assert set(dims["dimensions"]) == {"64", "128", "256"}
    assert all(math.isfinite(dims["dimensions"][key]["summary"]["mrr"]["mean"]) for key in dims["dimensions"])
    assert len(attribution["cases"]) == 5 and attribution["eligible_edges"] > 0
    assert max(case["attribution_additivity_max_abs_error"] for case in attribution["cases"]) < 1e-5
    assert efficiency["hybrid_over_deepsets"]["relative_parameter_increase_percent"] < 1.0
    assert read("dimension_implementation_equivalence.json")["exact"]
    assert external["coverage"]["included_prescriptions"] == external["coverage"]["evaluation_queries"]
    assert external["coverage"]["included_patients"] > 0
    assert external["overall"]["hybrid"]["model_seeds"] == 3
    assert external["overall"]["hybrid"]["mrr"]["mean"] > external["uniform_random_expectation"]["metrics"]["mrr"]
    assert external["coverage"]["herb_record_exact_coverage"] == mapping["herbs"]["record_coverage"]
    multi = read("reviewer_multisplit_results.json")
    for split in multi["splits"].values():
        assert split["paircomp"]["summary"]["mrr"]["mean"] > split["mean"]["summary"]["mrr"]["mean"]
        assert split["paircomp"]["summary"]["mrr"]["mean"] > split["matched"]["summary"]["mrr"]["mean"]
    gate = read("gate_ablation_results.json")
    assert gate["models"]["learned_size_gate"]["summary"]["mrr"]["mean"] > gate["models"]["fixed_pair_gate"]["summary"]["mrr"]["mean"]
    transparent = read("evaluation_transparency.json")
    assert transparent["overall"]["paircomp"]["typed"]["mrr"]["mean"] > transparent["overall"]["mean_complex"]["typed"]["mrr"]["mean"]
    assert transparent["overall"]["paircomp"]["untyped"]["mrr"]["mean"] < transparent["overall"]["mean_complex"]["untyped"]["mrr"]["mean"]
    set_ci = read("set_transformer_clustered_comparison.json")["paircomp_vs_set_transformer"]["cluster_bootstrap_95ci"]
    assert set_ci[0] > 0
    global_neg = read("global_negative_training_results.json")
    assert global_neg["models"]["paircomp"]["summary"]["untyped"]["mrr"]["mean"] > 0.4
    assert global_neg["models"]["mean_complex"]["summary"]["untyped"]["mrr"]["mean"] > 0.4
    global_ci = read("global_negative_clustered_comparison.json")
    mean_ci = global_ci["paircomp_vs_mean_complex"]["cluster_bootstrap_95ci"]
    matched_ci = global_ci["paircomp_vs_matched"]["cluster_bootstrap_95ci"]
    assert mean_ci[0] <= 0 <= mean_ci[1]
    assert matched_ci[0] > 0
    occlusion = read("pair_attribution_occlusion_sanity.json")
    assert occlusion["top_minus_random_formula_clustered_95ci"][0] > 0
    assert (HERE/"figures"/"paper"/"figure_8_review_evidence_abc.pdf").exists()
    ext_ci = external["patient_cluster_bootstrap"]["hybrid_vs_parameter_control"]["mrr"]["patient_cluster_bootstrap_95ci"]
    assert ext_ci[0] <= 0 <= ext_ci[1]  # external result is explicitly inconclusive
    external_zip = HERE / "external_data" / "ZhaoHanqing_TCM_Dataset_2025_v1.1.zip"
    assert hashlib.md5(external_zip.read_bytes()).hexdigest() == "7887afc67e8a9ffaf1f05af185773302"
    expected_figures = {
        "figure_model_architecture", "figure_results_and_size", "figure_robustness", "figure_transfer_and_dimension",
        "figure_external_validation"
    }
    assert all((HERE / "figures" / f"{stem}.{suffix}").exists() for stem in expected_figures for suffix in ("pdf", "png"))
    print("All Expert Systems artifact checks passed.")


if __name__ == "__main__":
    main()
