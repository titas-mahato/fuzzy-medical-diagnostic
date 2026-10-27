"""
Interactive Clinical Decision Support System (CDSS) Dashboard.
Course: CSE315 - Artificial Intelligence and Soft Computing (AISC)
Demonstrates Intelligent Agents, A* Search, ANN Backprop, Fuzzy FIS, GA & PSO.
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
        color: #2563EB;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-bottom: 1.5rem;
    }
    /* Theme-adaptive card compatible with light and dark mode */
    .metric-card {
        background-color: rgba(125, 125, 125, 0.08) !important;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid rgba(125, 125, 125, 0.2);
        border-left: 5px solid #3B82F6 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .metric-card h1, .metric-card h2, .metric-card h3, .metric-card h4 {
        margin-top: 4px;
        margin-bottom: 6px;
    }
    .metric-card p {
        margin-bottom: 4px;
        opacity: 0.9;
        font-size: 0.92rem;
    }
    .metric-card b {
        font-weight: 700;
    }
    .badge-mod {
        display: inline-block;
        background-color: rgba(59, 130, 246, 0.15);
        color: #3B82F6 !important;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .risk-low {
        color: #10B981 !important;
        font-weight: 800;
    }
    .risk-mod {
        color: #F59E0B !important;
        font-weight: 800;
    }
    .risk-high {
        color: #EF4444 !important;
        font-weight: 800;
    }
    .risk-crit {
        color: #DC2626 !important;
        font-weight: 900;
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
            <span class="badge-mod">PEAS Agent</span>
            <span class="badge-mod">Informed A* Search</span>
            <span class="badge-mod">Backprop MLP</span>
            <span class="badge-mod">Mamdani FIS</span>
            <span class="badge-mod">Clinical Database</span>
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
    "📊 Tab 4: Clinical Database & System Documentation",
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
                <span class="badge-mod">Mamdani FIS Engine</span>
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
                <span class="badge-mod">Feedforward Neural Network</span>
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
                <span class="badge-mod">Utility-Based Agent</span>
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

    with st.expander("📊 View Backpropagation Neural Network Evaluation Metrics"):
        m_c1, m_c2, m_c3, m_c4, m_c5, m_c6 = st.columns(6)
        m_c1.metric("Accuracy", f"{eval_metrics['accuracy']*100:.1f}%")
        m_c2.metric("Sensitivity (Recall)", f"{eval_metrics['recall']*100:.1f}%")
        m_c3.metric("Precision", f"{eval_metrics['precision']*100:.1f}%")
        m_c4.metric("Specificity", f"{eval_metrics['specificity']*100:.1f}%")
        m_c5.metric("F1-Score", f"{eval_metrics['f1_score']:.3f}")
        m_c6.metric("ROC-AUC", f"{eval_metrics['roc_auc']:.3f}")
        st.caption("Detailed 2x2 confusion matrix heatmap and epoch loss curves are available in Tab 4 (Model Evaluation).")

    # Soft Computing vs Hard Computing Noise Sensitivity Demonstration
    st.divider()
    st.markdown("#### 🔬 Soft vs. Hard Computing: Sensor Noise Robustness Experiment")
    st.caption("Evaluating resilience to physiological uncertainty, imprecision, and sensor noise.")

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
    st.markdown("#### Mamdani Fuzzy Inference System Visualizer")
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
        method_code = "centroid" if "Centroid" in defuzz_mode else ("mom" if "MOM" in defuzz_mode else "bisector")
        eval_res = default_fis.compute(patient_vitals, defuzz_method=method_code)

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
        fig, ax = plt.subplots(figsize=(8.5, 4.0))

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
            # Output Variable: Cardiac Risk Implication & Defuzzification
            z_u = eval_res["z_universe"]
            agg = eval_res["aggregated_curve"]
            term_alphas = eval_res.get("term_activations", {})

            term_colors = {
                "Low": "#2563EB",
                "Moderate": "#10B981",
                "High": "#F59E0B",
                "Critical": "#DC2626",
            }

            # 1. Unclipped consequent membership functions
            for t_name, t_mf in default_fis.output.terms.items():
                orig_mu = t_mf.evaluate(z_u)
                ax.plot(
                    z_u,
                    orig_mu,
                    linestyle="--",
                    linewidth=1.6,
                    color=term_colors.get(t_name, "#6B7280"),
                    label=f'Consequent "{t_name}"',
                    alpha=0.7,
                )

            # 2. Implied clipped area for patient
            ax.fill_between(z_u, 0, agg, color="#93C5FD", alpha=0.45, label="Implied Area (Mamdani Min)")
            ax.plot(z_u, agg, color="#1D4ED8", linewidth=2.5, label="Aggregated Consequent mu_agg(z)")

            # 3. Defuzzified crisp line
            z_star = eval_res["risk_score"]
            alg_name = defuzz_mode.split()[0]
            ax.axvline(
                z_star,
                color="#DC2626",
                linestyle="-",
                linewidth=2.5,
                label=f"{alg_name} Defuzzified z* = {z_star:.1f}%",
            )

            ax.set_title("Aggregated Mamdani Consequent & Patient Defuzzification", fontsize=12)
            ax.set_xlabel("Cardiovascular Risk Score (%)")
            ax.set_ylabel("Membership Degree mu(z)")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", fontsize=8.5)
            st.pyplot(fig)
            plt.close(fig)

    # Active Fired AHA Clinical Rules Table
    with st.expander("View Active Fired AHA Clinical Rules for this Patient", expanded=False):
        if eval_res["rule_activations"]:
            rule_df = pd.DataFrame(eval_res["rule_activations"])
            rule_df.columns = ["Clinical Rule (IF-THEN)", "Firing Strength (alpha)", "Consequent Term", "Clinical Rationale"]
            st.dataframe(rule_df, use_container_width=True, hide_index=True)
        else:
            st.info("No clinical rules fired above threshold for this patient.")

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
    st.markdown("#### Informed A* Diagnostic Test Pathway Search")
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
            border = "border-left: 5px solid #10B981 !important;" if is_astar else "border-left: 5px solid #6B7280 !important;"
            st.markdown(
                f"""
                <div class="metric-card" style="{border}">
                    <h4>{res['algorithm']}</h4>
                    <p>Financial Cost: <b>&#36;{res['total_dollar_cost']:.0f}</b></p>
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
# TAB 4: CLINICAL DATABASE & SYSTEM DOCUMENTATION
# ==============================================================================
with tab4:
    st.markdown("#### Clinical Database & System Documentation")
    st.caption("Explore the underlying UCI Heart Disease cohort, feature distributions, and the multi-stage AI diagnostic architecture.")

    sub_db1, sub_db2, sub_db3 = st.tabs([
        "📁 Clinical Dataset Explorer",
        "🔬 Model Evaluation & Risk Analysis",
        "📖 Architecture & Clinical Guidelines",
    ])

    with sub_db1:
        st.markdown("##### 🗄️ UCI Cleveland Heart Disease Cohort Analytics")

        # 4 Metric Cards for Dataset Stats
        total_records = len(heart_df)
        total_features = heart_df.shape[1] - 1
        pos_cases = int(np.sum(heart_df["target"] == 1))
        neg_cases = int(np.sum(heart_df["target"] == 0))
        pos_pct = (pos_cases / total_records) * 100.0
        neg_pct = (neg_cases / total_records) * 100.0

        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        with d_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span class="badge-mod">Sample Size</span>
                    <h2 style="margin:4px 0;">{total_records}</h2>
                    <p>Verified Patient Records</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with d_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span class="badge-mod">Biomarkers</span>
                    <h2 style="margin:4px 0;">{total_features} Attributes</h2>
                    <p>Clinical & Laboratory Features</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with d_col3:
            st.markdown(
                f"""
                <div class="metric-card" style="border-left: 5px solid #EF4444 !important;">
                    <span class="badge-mod" style="color:#EF4444 !important; border-color:rgba(239,68,68,0.3);">Confirmed CAD</span>
                    <h2 style="margin:4px 0; color:#EF4444;">{pos_cases} ({pos_pct:.1f}%)</h2>
                    <p>Ischemia / Stenosis Present</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with d_col4:
            st.markdown(
                f"""
                <div class="metric-card" style="border-left: 5px solid #10B981 !important;">
                    <span class="badge-mod" style="color:#10B981 !important; border-color:rgba(16,185,129,0.3);">Healthy Control</span>
                    <h2 style="margin:4px 0; color:#10B981;">{neg_cases} ({neg_pct:.1f}%)</h2>
                    <p>Absence of Vascular Disease</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # Interactive Filtering Controls
        st.markdown("##### 🔍 Interactive Cohort Filtering & Record Inspection")
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            diag_filter = st.selectbox("Filter by Disease Status:", ["All Records", "Heart Disease Present (Target=1)", "Healthy Controls (Target=0)"])
        with f_col2:
            min_age, max_age = int(heart_df["age"].min()), int(heart_df["age"].max())
            age_range = st.slider("Filter by Age Range:", min_age, max_age, (min_age, max_age))
        with f_col3:
            cp_filter = st.multiselect("Chest Pain Type (cp):", options=[0, 1, 2, 3], default=[0, 1, 2, 3], format_func=lambda x: {0: "Typical Angina (0)", 1: "Atypical Angina (1)", 2: "Non-Anginal (2)", 3: "Asymptomatic (3)"}[x])

        filtered_df = heart_df.copy()
        if diag_filter == "Heart Disease Present (Target=1)":
            filtered_df = filtered_df[filtered_df["target"] == 1]
        elif diag_filter == "Healthy Controls (Target=0)":
            filtered_df = filtered_df[filtered_df["target"] == 0]

        filtered_df = filtered_df[(filtered_df["age"] >= age_range[0]) & (filtered_df["age"] <= age_range[1])]
        if cp_filter:
            filtered_df = filtered_df[filtered_df["cp"].isin(cp_filter)]

        st.write(f"Displaying **{len(filtered_df)}** of {len(heart_df)} patient records matching criteria:")
        st.dataframe(filtered_df, use_container_width=True, height=240)

        # Matplotlib Distribution Analytics
        st.markdown("##### 📈 Cohort Biomarker Distribution & Clustering")
        ch_col1, ch_col2 = st.columns(2)

        with ch_col1:
            fig_sc, ax_sc = plt.subplots(figsize=(6, 3.8))
            healthy_mask = heart_df["target"] == 0
            disease_mask = heart_df["target"] == 1

            ax_sc.scatter(heart_df.loc[healthy_mask, "trestbps"], heart_df.loc[healthy_mask, "chol"], color="#10B981", alpha=0.6, label="Healthy (Target=0)", edgecolors="none", s=35)
            ax_sc.scatter(heart_df.loc[disease_mask, "trestbps"], heart_df.loc[disease_mask, "chol"], color="#EF4444", alpha=0.7, label="CAD Disease (Target=1)", edgecolors="none", s=35)

            # Clinical Guideline Thresholds
            ax_sc.axvline(140, color="#D97706", linestyle="--", linewidth=1.2, label="Hypertension Cutoff (140 mmHg)")
            ax_sc.axhline(240, color="#7C3AED", linestyle="--", linewidth=1.2, label="High Chol Cutoff (240 mg/dL)")

            ax_sc.set_title("Resting BP vs. Serum Cholesterol Clustering", fontsize=11)
            ax_sc.set_xlabel("Resting Blood Pressure (mmHg)")
            ax_sc.set_ylabel("Serum Cholesterol (mg/dL)")
            ax_sc.legend(fontsize=8, loc="upper right")
            ax_sc.grid(True, linestyle=":", alpha=0.5)
            st.pyplot(fig_sc)
            plt.close(fig_sc)

        with ch_col2:
            fig_hist, ax_h = plt.subplots(figsize=(6, 3.8))
            ax_h.hist(heart_df.loc[healthy_mask, "age"], bins=15, alpha=0.5, color="#10B981", label="Healthy Controls", density=True)
            ax_h.hist(heart_df.loc[disease_mask, "age"], bins=15, alpha=0.6, color="#EF4444", label="Disease Cohort", density=True)

            ax_h.set_title("Age Density Distribution by Pathology", fontsize=11)
            ax_h.set_xlabel("Patient Age (years)")
            ax_h.set_ylabel("Relative Density")
            ax_h.legend(fontsize=8, loc="upper right")
            ax_h.grid(True, linestyle=":", alpha=0.5)
            st.pyplot(fig_hist)
            plt.close(fig_hist)

        with st.expander("📊 Full Statistical Summary (Mean, Std Dev, Min/Max)"):
            st.dataframe(heart_df.describe().T, use_container_width=True)

    with sub_db2:
        st.markdown("##### 🔬 Neural Network Classification & Clinical Risk Analysis")
        st.caption("Quantitative evaluation of the 3-Layer Multi-Layer Perceptron trained with Supervised Backpropagation and the Delta Learning Rule.")

        # 6 Evaluation Metric Badges
        cm = eval_metrics["confusion_matrix"]
        tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]

        r_col1, r_col2, r_col3, r_col4, r_col5, r_col6 = st.columns(6)
        with r_col1:
            st.metric("Test Accuracy", f"{eval_metrics['accuracy']*100:.1f}%")
        with r_col2:
            st.metric("Precision (PPV)", f"{eval_metrics['precision']*100:.1f}%")
        with r_col3:
            st.metric("Recall / Sensitivity", f"{eval_metrics['recall']*100:.1f}%", help="True Positive Rate: Critical for detecting disease and avoiding missed diagnoses.")
        with r_col4:
            st.metric("Specificity (TNR)", f"{eval_metrics['specificity']*100:.1f}%", help="True Negative Rate: Correctly identifying healthy individuals.")
        with r_col5:
            st.metric("F1-Score", f"{eval_metrics['f1_score']:.3f}", help="Harmonic mean of precision and recall.")
        with r_col6:
            st.metric("ROC-AUC", f"{eval_metrics['roc_auc']:.3f}", help="Area Under ROC Curve: Discrimination ability between diseased and healthy cases.")

        st.divider()

        # Two Charts: Confusion Matrix Heatmap & Backpropagation Learning Loss Curves
        ev_c1, ev_c2 = st.columns(2)

        with ev_c1:
            st.markdown("###### Confusion Matrix Heatmap")
            fig_cm, ax_cm = plt.subplots(figsize=(5, 3.8))
            cm_matrix = np.array([[tn, fp], [fn, tp]])
            cax = ax_cm.matshow(cm_matrix, cmap="Blues", alpha=0.85)

            # Labels and annotations
            for i in range(2):
                for j in range(2):
                    val = cm_matrix[i, j]
                    label = f"{val}\n({'TN' if (i==0 and j==0) else 'FP' if (i==0 and j==1) else 'FN' if (i==1 and j==0) else 'TP'})"
                    color = "white" if val > cm_matrix.max() / 2 else "black"
                    ax_cm.text(j, i, label, ha="center", va="center", color=color, fontsize=12, fontweight="bold")

            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(["Predicted Healthy (0)", "Predicted Disease (1)"], fontsize=9)
            ax_cm.set_yticklabels(["Actual Healthy (0)", "Actual Disease (1)"], fontsize=9)
            ax_cm.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
            ax_cm.set_ylabel("True Ground-Truth", fontsize=10, labelpad=8)
            ax_cm.set_title("Test Set Confusion Matrix", fontsize=11, pad=12)
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with ev_c2:
            st.markdown("###### Backpropagation Training & Validation Loss")
            if mlp_model.training_history:
                hist_df = pd.DataFrame(mlp_model.training_history)
                fig_loss, ax_l = plt.subplots(figsize=(5.5, 3.8))
                ax_l.plot(hist_df["epoch"], hist_df["train_loss"], color="#2563EB", label="Train BCE Loss", linewidth=1.8)
                if "val_loss" in hist_df.columns:
                    ax_l.plot(hist_df["epoch"], hist_df["val_loss"], color="#EF4444", linestyle="--", label="Val BCE Loss", linewidth=1.8)
                ax_l.set_title("Supervised Learning Convergence (Delta Rule)", fontsize=11)
                ax_l.set_xlabel("Training Epochs", fontsize=10)
                ax_l.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
                ax_l.legend(fontsize=9, loc="upper right")
                ax_l.grid(True, linestyle=":", alpha=0.6)
                st.pyplot(fig_loss)
                plt.close(fig_loss)

        st.markdown("###### Clinical Diagnostic Risk Analysis")
        st.info(
            f"**Clinical Interpretation of Trade-Offs:**\n\n"
            f"- **Sensitivity / Recall ({eval_metrics['recall']*100:.1f}%):** In cardiovascular triage, minimizing False Negatives (FN = {fn}) is paramount. A false negative means an ischemic patient is discharged without treatment.\n"
            f"- **Precision ({eval_metrics['precision']*100:.1f}%):** Minimizes False Positives (FP = {fp}), preventing unnecessary expensive follow-up interventions (e.g. catheter angiography).\n"
            f"- **ROC-AUC ({eval_metrics['roc_auc']:.3f}):** Indicates strong clinical discrimination power between pathological coronary stenosis and healthy hemodynamic controls."
        )

    with sub_db3:
        st.markdown("##### 🏛️ System Architecture & Clinical Knowledge Base")

        st.markdown(
            """
            This clinical decision support system integrates multi-paradigm soft computing and search algorithms:
            1. **Sensor Ingestion:** Ingests hemodynamic vital signs and laboratory blood assays.
            2. **Fuzzy Mamdani Inference:** Evaluates non-linear clinical uncertainty through linguistic IF-THEN rules and calculates a continuous risk score via Centroid defuzzification.
            3. **Neural Network Classification:** Uses a 3-layer backpropagation MLP trained on patient records to generate an independent posterior probability.
            4. **Utility Decision Engine:** A rational PEAS agent balances diagnostic confidence against financial expense and patient discomfort.
            5. **Informed A* Diagnostic Pathway:** If further clinical clarity is required, an admissible Shannon entropy heuristic guides cost-optimal test sequencing.
            """
        )

        st.divider()

        st.markdown("##### 🩺 Clinical Practice Guidelines & Diagnostic Cutoffs")
        guidelines_data = [
            {"Vital Parameter": "Blood Pressure (Systolic)", "Optimal / Normal": "< 120 mmHg", "Borderline / Pre-Hypertension": "120 - 139 mmHg", "High Risk / Hypertension": ">= 140 mmHg (Crisis: >= 180)", "Guideline Authority": "AHA / ACC 2017"},
            {"Vital Parameter": "Total Cholesterol", "Optimal / Normal": "< 200 mg/dL", "Borderline / Pre-Hypertension": "200 - 239 mg/dL", "High Risk / Hypertension": ">= 240 mg/dL", "Guideline Authority": "NCEP ATP III"},
            {"Vital Parameter": "Fasting Blood Sugar", "Optimal / Normal": "< 100 mg/dL", "Borderline / Pre-Hypertension": "100 - 125 mg/dL", "High Risk / Hypertension": ">= 126 mg/dL (Diabetes)", "Guideline Authority": "American Diabetes Assoc. (ADA)"},
            {"Vital Parameter": "Maximum Heart Rate", "Optimal / Normal": "Age-appropriate (220 - Age)", "Borderline / Pre-Hypertension": "Blunted chronotropic response", "High Risk / Hypertension": "Severe tachycardia / bradycardia", "Guideline Authority": "Framingham Heart Study"},
        ]
        st.dataframe(pd.DataFrame(guidelines_data), use_container_width=True, hide_index=True)

        st.divider()

        st.markdown("##### 🧪 Diagnostic Investigation Catalog & Decision Metrics")
        catalog_data = []
        for test_key, test_obj in search_engine.test_catalog.items():
            catalog_data.append({
                "Diagnostic Procedure": test_obj.name,
                "Financial Cost": f"${test_obj.cost_dollars:.0f}",
                "Discomfort Index (1-10)": f"{test_obj.discomfort_score:.1f}",
                "Diagnostic Information Gain": f"{test_obj.entropy_reduction * 100:.0f}%",
                "Clinical Indication": test_obj.description,
            })
        st.dataframe(pd.DataFrame(catalog_data), use_container_width=True, hide_index=True)

        st.divider()

        st.markdown("##### 📐 Mathematical Formulations Summary")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(
                """
                **Mamdani Centroid Defuzzification:**
                $$z^* = \\frac{\\sum_{i=1}^M z_i \\cdot \\mu_{\\text{agg}}(z_i)}{\\sum_{i=1}^M \\mu_{\\text{agg}}(z_i)}$$
                
                **A\\* Admissible Entropy Heuristic:**
                $$h(n) = \\frac{\\max(0, H(n) - \\tau)}{\\epsilon_{\\max}}, \\quad \\epsilon_{\\max} = \\max_t \\left( \\frac{\\Delta H_t}{c(t)} \\right)$$
                """
            )
        with m_col2:
            st.markdown(
                """
                **Supervised Backpropagation Delta Rule:**
                $$\\delta^{[3]} = \\hat{y} - y$$
                $$\\delta^{[l]} = \\left( \\delta^{[l+1]} (W^{[l+1]})^T \\right) \\odot \\text{ReLU}'(z^{[l]})$$
                
                **Multi-Attribute Clinical Utility:**
                $$U(a) = w_1 \\cdot \\text{Certainty} - w_2 \\cdot \\text{Cost} - w_3 \\cdot \\text{Invasiveness}$$
                """
            )

st.markdown("---")
st.caption("CardioVision CDSS v1.0 | Intelligent Clinical Decision Support System.")
