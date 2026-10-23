"""
Interactive Clinical Decision Support System (CDSS) Dashboard.
Course: CSE315 - Artificial Intelligence and Soft Computing (AISC)
Demonstrates Modules 1-5: Intelligent Agents, A* Search, ANN Backprop, Fuzzy FIS, GA & PSO.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.fuzzy.mamdani_fis import MamdaniFIS
from src.fuzzy.sugeno_fis import SugenoFIS
from src.models.mlp_classifier import train_mlp_on_dataset
from src.search.astar_pathway import ClinicalPathwaySearch
from src.agents.diagnostic_agent import (
    UtilityBasedDiagnosticAgent,
    GoalBasedDiagnosticAgent,
    SoftVsHardComputingBenchmark,
)
from src.optimization.genetic_selector import GeneticFeatureSelector
from src.optimization.pso_tuner import ParticleSwarmTuner

# Page Configuration
st.set_page_config(
    page_title="CardioVision CDSS | CSE315 AISC",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Medical Aesthetic Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .badge-mod {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #BFDBFE;
    }
    .risk-low {
        color: #059669;
        font-weight: bold;
    }
    .risk-mod {
        color: #D97706;
        font-weight: bold;
    }
    .risk-high {
        color: #DC2626;
        font-weight: bold;
    }
    .risk-crit {
        color: #7F1D1D;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training Backpropagation MLP Model...")
def load_system_resources():
    csv_path = str(PROJECT_ROOT / "data" / "heart.csv")
    mlp_model, eval_metrics, df = train_mlp_on_dataset(csv_path=csv_path, epochs=150)
    fuzzy_fis = MamdaniFIS()
    search_engine = ClinicalPathwaySearch()
    agent = UtilityBasedDiagnosticAgent()
    return mlp_model, eval_metrics, df, fuzzy_fis, search_engine, agent


mlp_model, eval_metrics, heart_df, default_fis, search_engine, agent = load_system_resources()

# Sidebar: Course Badges & Presets
with st.sidebar:
    st.markdown("### 🩺 CardioVision CDSS")
    st.caption("**CSE315: Artificial Intelligence & Soft Computing**")
    st.markdown(
        """
        <div style="display:flex; flex-wrap:wrap; gap:5px; margin-bottom:15px;">
            <span class="badge-mod">Mod 1: PEAS Agent</span>
            <span class="badge-mod">Mod 2: A* Search</span>
            <span class="badge-mod">Mod 3: MLP Backprop</span>
            <span class="badge-mod">Mod 4: Mamdani FIS</span>
            <span class="badge-mod">Mod 5: GA & PSO</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("📋 Patient Clinical Profiles")
    preset = st.selectbox(
        "Load Clinical Case Preset:",
        [
            "Custom Patient (Interactive)",
            "Case A: Healthy Young Athlete (Low Risk)",
            "Case B: Borderline Metabolic (Moderate Risk)",
            "Case C: Stage-2 Hypertensive Diabetic (High Risk)",
            "Case D: Critical Hypertensive Emergency (Critical)",
        ],
    )

    # Preset values
    if preset == "Case A: Healthy Young Athlete (Low Risk)":
        def_bp, def_chol, def_glu, def_hr, def_age = 110, 155, 82, 160, 26
    elif preset == "Case B: Borderline Metabolic (Moderate Risk)":
        def_bp, def_chol, def_glu, def_hr, def_age = 135, 225, 115, 140, 52
    elif preset == "Case C: Stage-2 Hypertensive Diabetic (High Risk)":
        def_bp, def_chol, def_glu, def_hr, def_age = 158, 255, 145, 125, 66
    elif preset == "Case D: Critical Hypertensive Emergency (Critical)":
        def_bp, def_chol, def_glu, def_hr, def_age = 190, 310, 180, 175, 72
    else:
        def_bp, def_chol, def_glu, def_hr, def_age = 142, 238, 118, 145, 58

    st.divider()
    st.markdown("**Local Dataset Status:**")
    st.write(f"✓ UCI Heart Data Loaded: `{len(heart_df)} records`")
    st.write(f"✓ MLP Test Accuracy: `{eval_metrics['accuracy']*100:.1f}%`")
    st.write(f"✓ MLP Test ROC-AUC: `{eval_metrics['roc_auc']:.3f}`")

# Header Section
st.markdown('<div class="main-header">Cardiovascular Clinical Decision Support System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Rigorous Soft Computing & Intelligent Search Architecture for Academic Clinical Risk Triage</div>',
    unsafe_allow_html=True,
)

# Main Application Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Tab 1: Patient Triage & PEAS Agent",
    "📐 Tab 2: Fuzzy Membership & Inference Visualizer",
    "🧭 Tab 3: A* Intelligent Test Pathway Search",
    "🧬 Tab 4: Evolutionary (GA) & Swarm (PSO) Convergence",
])

# ==============================================================================
# TAB 1: PATIENT DIAGNOSIS & PEAS AGENT
# ==============================================================================
with tab1:
    st.markdown("#### Patient Clinical Hemodynamics & Biomarkers")

    col_inp1, col_inp2, col_inp3, col_inp4, col_inp5 = st.columns(5)
    with col_inp1:
        val_bp = st.slider("Blood Pressure (trestbps, mmHg)", 70, 220, int(def_bp), help="Resting systolic blood pressure.")
    with col_inp2:
        val_chol = st.slider("Cholesterol (chol, mg/dL)", 100, 400, int(def_chol), help="Serum total cholesterol.")
    with col_inp3:
        val_glu = st.slider("Fasting Blood Sugar (fbs, mg/dL)", 60, 280, int(def_glu), help="Fasting blood glucose.")
    with col_inp4:
        val_hr = st.slider("Max Heart Rate (thalach, bpm)", 50, 220, int(def_hr), help="Maximum heart rate achieved.")
    with col_inp5:
        val_age = st.slider("Patient Age (years)", 20, 95, int(def_age), help="Patient chronological age.")

    patient_vitals = {
        "blood_pressure": float(val_bp),
        "cholesterol": float(val_chol),
        "glucose": float(val_glu),
        "heart_rate": float(val_hr),
        "age": float(val_age),
    }

    # Execute Diagnostic Engines
    fuzzy_diag = default_fis.compute(patient_vitals)
    ann_feat = np.array([[float(val_age), float(val_bp), float(val_chol), 1.0 if val_glu > 120 else 0.0, float(val_hr)]])
    mlp_proba = float(mlp_model.predict_proba(ann_feat)[0]) * 100.0
    action = agent.diagnose_and_act(patient_vitals, mlp_model)

    st.divider()

    # Three Clinical Diagnostic Metric Cards
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        risk_color = (
            "risk-low" if fuzzy_diag["risk_score"] < 25
            else "risk-mod" if fuzzy_diag["risk_score"] < 55
            else "risk-high" if fuzzy_diag["risk_score"] < 80
            else "risk-crit"
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <span class="badge-mod">Module 4: Mamdani FIS</span>
                <h3 style="margin-top:8px;">Fuzzy Risk Score</h3>
                <h1 class="{risk_color}">{fuzzy_diag['risk_score']:.1f}%</h1>
                <p>Category: <b>{fuzzy_diag['risk_category']}</b> (Centroid Defuzzification)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        ann_color = "risk-low" if mlp_proba < 35 else "risk-mod" if mlp_proba < 65 else "risk-high"
        st.markdown(
            f"""
            <div class="metric-card">
                <span class="badge-mod">Module 3: Neural Network</span>
                <h3 style="margin-top:8px;">MLP Disease Probability</h3>
                <h1 class="{ann_color}">{mlp_proba:.1f}%</h1>
                <p>3-Layer Backprop Model (Delta Rule, 16x8 hidden neurons)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <span class="badge-mod">Module 1: Utility Agent</span>
                <h3 style="margin-top:8px;">Expected Utility EU(a)</h3>
                <h1 style="color:#2563EB;">{action.expected_utility:.3f}</h1>
                <p>Action: <b>{action.triage_tier.split(':')[0]}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Agent Actionable Clinical Recommendation Box
    st.markdown("#### 🏥 PEAS Clinical Recommendation & Prescriptive Advice")
    triage_alert = st.success if "Tier 1" in action.triage_tier else st.info if "Tier 2" in action.triage_tier else st.warning if "Tier 3" in action.triage_tier else st.error
    triage_alert(f"**{action.triage_tier}**\n\n{action.explanation}")

    with st.expander("🩺 Recommended Pharmacotherapy & Clinical Protocol"):
        for item in action.prescriptive_advice:
            st.markdown(f"- {item}")
        pathway_str = ' -> '.join(action.recommended_pathway)
        st.markdown(rf"- **A\* Recommended Diagnostic Test Sequence:** `{pathway_str}`")

    with st.expander("📜 Fired Mamdani Fuzzy Rules & Firing Strengths (alpha)"):
        if fuzzy_diag["rule_activations"]:
            rule_df = pd.DataFrame(fuzzy_diag["rule_activations"])
            rule_df.rename(columns={"rule": "Clinical Rule", "strength": "Firing Strength (alpha)", "consequent": "Consequent", "rationale": "Medical Rationale"}, inplace=True)
            st.dataframe(rule_df, use_container_width=True)
        else:
            st.info("Baseline default rules applied (no high-risk antecendent fired).")

    # Soft Computing vs Hard Computing Noise Sensitivity Demonstration
    st.divider()
    st.markdown("#### 🔬 Soft vs. Hard Computing: Sensor Noise Robustness Experiment")
    st.caption("Demonstrating handling of uncertainty, imprecision, and sensor noise (Syllabus Module 1).")

    noise_col1, noise_col2 = st.columns([1, 2])
    with noise_col1:
        test_noise = st.slider("Simulated Sphygmomanometer Noise (std dev in mmHg)", 1.0, 10.0, 3.5, 0.5)
        run_sim = st.button("Simulate 40 Sensor Readings")

    with noise_col2:
        if run_sim:
            sim_res = SoftVsHardComputingBenchmark.simulate_noise_robustness(patient_vitals, noise_std=test_noise, trials=40)
            st.write(f"**Hard (Crisp) Logic Category Flipping:** `{sim_res['hard_category_distribution']}`")
            st.write(f"**Soft (Fuzzy) Continuous Risk Output:** Mean = `{sim_res['soft_risk_mean']:.1f}%`, Std Dev = `±{sim_res['soft_risk_std']:.2f}%`")
            if sim_res["hard_instability_score"] > 1:
                st.warning("⚠️ **Hard Computing Instability Observed:** Crisp thresholding caused multiple discrete category flips under minor physiological fluctuation. Soft Fuzzy Computing preserved continuous, graded stability.")
            else:
                st.info("Inputs are sufficiently far from crisp thresholds; both models maintained stable categories.")


# ==============================================================================
# TAB 2: FUZZY MEMBERSHIP & INFERENCE VISUALIZER
# ==============================================================================
with tab2:
    st.markdown("#### Mamdani Fuzzy Inference System Visualizer (Module 4)")
    st.caption("Mathematical representation of Triangular & Trapezoidal Membership Functions, Implication Clipping, and Centroid Defuzzification.")

    col_mf1, col_mf2 = st.columns([1, 2])

    with col_mf1:
        selected_var = st.selectbox(
            "Select Linguistic Variable to Inspect:",
            ["blood_pressure", "cholesterol", "glucose", "heart_rate", "age", "cardiac_risk"],
        )
        defuzz_mode = st.selectbox(
            "Defuzzification Algorithm Comparison:",
            ["Centroid (Center of Gravity)", "Mean of Maxima (MOM)", "Bisector"],
        )

        st.markdown(
            """
            **Defuzzification Mathematical Formulations:**
            - **Centroid (COG):**
              $$z^* = \\frac{\\sum z_i \\cdot \\mu_{agg}(z_i)}{\\sum \\mu_{agg}(z_i)}$$
            - **MOM:**
              $$z^* = \\frac{1}{|M|} \\sum_{z_i \\in M} z_i$$
            - **Bisector:**
              $$\\int_{-\\infty}^{z^*} \\mu(z)dz = \\int_{z^*}^{\\infty} \\mu(z)dz$$
            """
        )

    with col_mf2:
        fig, ax = plt.subplots(figsize=(8, 3.8))

        if selected_var in default_fis.inputs:
            var_obj = default_fis.inputs[selected_var]
            u_x = var_obj.universe
            patient_val = patient_vitals.get(selected_var, (var_obj.min_val + var_obj.max_val) / 2)

            for term_name, mf in var_obj.terms.items():
                mu_curve = mf.evaluate(u_x)
                ax.plot(u_x, mu_curve, label=term_name, linewidth=2)

            ax.axvline(patient_val, color="red", linestyle="--", linewidth=2, label=f"Patient Input ({patient_val:.1f})")
            ax.set_title(f"Membership Functions for {selected_var.replace('_', ' ').title()}", fontsize=12)
            ax.set_xlabel(selected_var)
            ax.set_ylabel("Degree of Membership (mu)")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right")
            st.pyplot(fig)
            plt.close(fig)

        else:
            # Output Variable: Cardiac Risk Defuzzification Curve
            z_u = fuzzy_diag["z_universe"]
            agg = fuzzy_diag["aggregated_curve"]
            fig_defuzz, ax_d = plt.subplots(figsize=(8, 3.8))

            ax_d.plot(z_u, agg, color="#1D4ED8", linewidth=2.5, label="Aggregated Consequent mu_agg(z)")
            ax_d.fill_between(z_u, 0, agg, color="#93C5FD", alpha=0.45, label="Implied Area (Mamdani Min)")

            cog_z = fuzzy_diag["risk_score"]
            ax_d.axvline(cog_z, color="#DC2626", linestyle="-", linewidth=2.5, label=f"Centroid Defuzzified z* = {cog_z:.1f}%")

            ax_d.set_title("Aggregated Mamdani Consequent & Centroid Defuzzification", fontsize=12)
            ax_d.set_xlabel("Cardiovascular Risk Score (%)")
            ax_d.set_ylabel("Aggregated Membership mu_agg")
            ax_d.set_ylim(-0.05, 1.05)
            ax_d.grid(True, linestyle=":", alpha=0.6)
            ax_d.legend(loc="upper right")
            st.pyplot(fig_defuzz)
            plt.close(fig_defuzz)

    st.divider()

    # Sugeno FIS Comparison & ANFIS Table
    st.markdown("#### Mamdani vs. Sugeno FIS & ANFIS Architecture Matrix")
    col_comp1, col_comp2 = st.columns(2)

    with col_comp1:
        sugeno_engine = SugenoFIS(order=1)
        fuzzified_inp = default_fis.fuzzify_all(patient_vitals)
        sugeno_res = sugeno_engine.compute(patient_vitals, fuzzified_inp)

        st.markdown(
            f"""
            <div class="metric-card">
                <h4>Sugeno (TSK) 1st-Order Inference</h4>
                <p>Linear Consequent: <code>z = p0 + p1*BP + p2*Chol + p3*Age</code></p>
                <h3>Sugeno Risk Output: <span style="color:#2563EB;">{sugeno_res['risk_score']:.1f}%</span></h3>
                <p>Weighted Average Defuzzification: <code>z* = sum(w_i * z_i) / sum(w_i)</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_comp2:
        anfis_dict = SugenoFIS.anfis_architecture_breakdown()
        anfis_df = pd.DataFrame(list(anfis_dict.items()), columns=["ANFIS Layer", "Functional Architecture"])
        st.dataframe(anfis_df, use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 3: A* INTELLIGENT TEST PATHWAY SEARCH
# ==============================================================================
with tab3:
    st.markdown("#### Module 2: Informed A* Diagnostic Test Pathway Search")
    st.caption("State-space heuristic search balancing diagnostic certainty against financial cost and patient invasiveness.")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        target_entropy = st.slider("Target Diagnostic Entropy threshold (tau)", 0.05, 0.40, 0.20, 0.05, help="Lower entropy means higher certainty requirement.")
    with col_s2:
        patient_prior = st.slider("Prior Patient Disease Probability", 0.10, 0.90, float(action.risk_score / 100.0), 0.05)
    with col_s3:
        budget_limit = st.number_input("Maximum Financial Budget (\\$)", min_value=100.0, max_value=3500.0, value=2500.0, step=100.0)

    search_engine.max_budget = budget_limit
    search_results = search_engine.compare_search_algorithms(target_entropy=target_entropy, prior=patient_prior)

    # Side-by-side search algorithm comparison cards
    col_a1, col_a2, col_a3 = st.columns(3)
    alg_cards = [col_a1, col_a2, col_a3]

    for idx, res in enumerate(search_results):
        with alg_cards[idx]:
            is_astar = "A*" in res["algorithm"]
            border = "border-left: 5px solid #10B981;" if is_astar else "border-left: 5px solid #6B7280;"
            st.markdown(
                f"""
                <div class="metric-card" style="{border}">
                    <h4>{res['algorithm']}</h4>
                    <p>Financial Cost: <b>\\${res['total_dollar_cost']:.0f}</b></p>
                    <p>Patient Invasiveness: <b>{res['total_discomfort']:.1f} / 10</b></p>
                    <p>Certainty Achieved: <b>{res['certainty_percentage']:.1f}%</b></p>
                    <p>Nodes Expanded: <b>{res['nodes_expanded']}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("##### 📍 Optimal Pathway Sequence Discovered by A* Search:")
    astar_res = search_results[0]
    if astar_res["path"]:
        st.info(" ➡️ ".join([f"**Step {i+1}:** {t}" for i, t in enumerate(astar_res["path"])]))
    else:
        st.warning("Diagnostic certainty already satisfies target; no further testing required.")

    with st.expander("🔍 Detailed A* Step-by-Step Search Tree Trace (f = g + h)"):
        trace_df = pd.DataFrame(astar_res["search_trace"])
        st.dataframe(trace_df, use_container_width=True)


# ==============================================================================
# TAB 4: EVOLUTIONARY & SWARM CONVERGENCE
# ==============================================================================
with tab4:
    st.markdown("#### Module 5: Evolutionary Algorithms & Swarm Intelligence")
    st.caption("Demonstrating Genetic Algorithm (GA) feature biomarker selection and Particle Swarm Optimization (PSO) membership tuning.")

    sub_opt1, sub_opt2 = st.tabs(["🧬 Genetic Algorithm (Biomarker Selection)", "🐝 Particle Swarm Optimization (PSO Tuner)"])

    with sub_opt1:
        ga_c1, ga_c2 = st.columns([1, 2])
        with ga_c1:
            st.markdown("##### GA Hyperparameters")
            ga_pop = st.slider("Population Size", 10, 40, 16, 2)
            ga_gens = st.slider("Generations", 5, 30, 12, 1)
            ga_pm = st.slider("Bit-Flip Mutation Rate (pm)", 0.01, 0.20, 0.06, 0.01)
            run_ga_btn = st.button("🚀 Run Genetic Feature Selection")

            st.caption(GeneticFeatureSelector.schema_theorem_summary())

        with ga_c2:
            if run_ga_btn:
                all_feats = [c for c in heart_df.columns if c != "target"]
                ga_engine = GeneticFeatureSelector(
                    population_size=ga_pop,
                    generations=ga_gens,
                    mutation_rate=ga_pm,
                    random_state=42,
                )
                with st.spinner("Executing Genetic Selection across generations..."):
                    ga_out = ga_engine.fit(heart_df[all_feats].values, heart_df["target"].values, all_feats)

                st.success(f"GA Converged! Selected **{ga_out['selected_count']}** out of {len(all_feats)} Biomarkers with **{ga_out['best_cv_accuracy']*100:.1f}%** 3-Fold Cross-Validation Accuracy.")
                st.write("**Optimal Biomarker Panel:**", ", ".join([f"`{f}`" for f in ga_out["selected_features"]]))

                # Plot GA convergence
                hist_df = pd.DataFrame(ga_out["history"])
                fig_ga, ax_g = plt.subplots(figsize=(7, 3.2))
                ax_g.plot(hist_df["generation"], hist_df["best_fitness"], marker="o", color="#1D4ED8", label="Best Fitness")
                ax_g.plot(hist_df["generation"], hist_df["avg_fitness"], linestyle="--", color="#93C5FD", label="Average Fitness")
                ax_g.set_title("Genetic Algorithm Fitness Convergence", fontsize=11)
                ax_g.set_xlabel("Generation")
                ax_g.set_ylabel("Multi-Objective Fitness")
                ax_g.grid(True, linestyle=":", alpha=0.6)
                ax_g.legend()
                st.pyplot(fig_ga)
                plt.close(fig_ga)
            else:
                st.info("Click 'Run Genetic Feature Selection' to start chromosome evolution.")

    with sub_opt2:
        pso_c1, pso_c2 = st.columns([1, 2])
        with pso_c1:
            st.markdown("##### Swarm Hyperparameters")
            pso_particles = st.slider("Swarm Size (Particles)", 6, 25, 12, 2)
            pso_iters = st.slider("PSO Iterations", 5, 25, 10, 1)
            pso_c1_coef = st.slider("Cognitive Coefficient (c1)", 0.5, 2.5, 1.8, 0.1)
            pso_c2_coef = st.slider("Social Coefficient (c2)", 0.5, 2.5, 1.8, 0.1)
            run_pso_btn = st.button("🐝 Run PSO Membership Tuner")

            st.caption("Velocity Update: `v_i(t+1) = w*v_i + c1*r1*(pBest - x) + c2*r2*(gBest - x)`")

        with pso_c2:
            if run_pso_btn:
                pso_engine = ParticleSwarmTuner(
                    num_particles=pso_particles,
                    max_iterations=pso_iters,
                    c1=pso_c1_coef,
                    c2=pso_c2_coef,
                    random_state=42,
                )
                with st.spinner("Swarm optimizing membership inflection parameters..."):
                    pso_out = pso_engine.fit(heart_df, sample_size=35)

                st.success(
                    f"PSO Optimization Complete! RMSE reduced from **{pso_out['baseline_rmse']:.2f}** down to **{pso_out['tuned_rmse']:.2f}** ({pso_out['improvement_pct']:.1f}% improvement)."
                )

                # Plot PSO convergence
                pso_hist = pd.DataFrame(pso_out["convergence_curve"])
                fig_pso, ax_p = plt.subplots(figsize=(7, 3.2))
                ax_p.plot(pso_hist["iteration"], pso_hist["gbest_rmse"], marker="s", color="#D97706", label="Global Best (gBest) RMSE")
                ax_p.set_title("Particle Swarm Optimization Convergence", fontsize=11)
                ax_p.set_xlabel("Iteration")
                ax_p.set_ylabel("Diagnostic RMSE Loss")
                ax_p.grid(True, linestyle=":", alpha=0.6)
                ax_p.legend()
                st.pyplot(fig_pso)
                plt.close(fig_pso)

                with st.expander("🔍 Tuned Membership Boundary Parameters"):
                    st.json(pso_out["tuned_parameters"])
            else:
                st.info("Click 'Run PSO Membership Tuner' to optimize fuzzy membership boundaries.")

st.markdown("---")
st.caption("CardioVision CDSS v1.0 | Academic project engineered for CSE315: Artificial Intelligence and Soft Computing.")
