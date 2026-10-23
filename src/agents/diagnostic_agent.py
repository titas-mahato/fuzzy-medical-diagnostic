"""
Goal-Based and Utility-Based Intelligent Clinical Agents.
Implements the PEAS Framework and Soft vs. Hard Computing Comparative Benchmark.
Course: CSE315 - Module 1: Introduction to AI and Soft Computing
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import numpy as np

from ..fuzzy.mamdani_fis import MamdaniFIS
from ..models.mlp_classifier import MLPClassifier
from ..search.astar_pathway import ClinicalPathwaySearch


@dataclass
class PEASSpecification:
    """Academic formalization of the PEAS framework for the CDSS Agent."""

    performance_measures: List[str] = field(
        default_factory=lambda: [
            "Diagnostic Certainty: Maximizing post-test probability confidence (1 - Shannon Entropy)",
            "Financial Prudence: Minimizing total monetary expense in test sequencing",
            "Patient Safety & Comfort: Minimizing invasiveness index (Scale 1-10)",
            "Clinical Timeliness: Minimizing time-to-diagnosis",
        ]
    )
    environment_properties: Dict[str, str] = field(
        default_factory=lambda: {
            "Observability": "Partially Observable (latent coronary atherosclerosis hidden beneath superficial vitals)",
            "Determinism": "Stochastic (patient biological variability, test assay noise, non-deterministic pathology)",
            "Episodic vs Sequential": "Sequential (each test result dictates the necessity and choice of the next investigation)",
            "Static vs Dynamic": "Dynamic (hemodynamic vitals and ischemia fluctuate continuously with stress)",
            "Discrete vs Continuous": "Continuous (blood pressure, blood glucose, and lipid titers are real continuous numbers)",
            "Agents": "Single-Agent (Diagnostic system reasoning against the patient biological environment)",
        }
    )
    actuators: List[str] = field(
        default_factory=lambda: [
            "Risk Category Assignment: Low, Moderate, High, or Critical risk stratification",
            "Optimal Test Order: Invoking A* Search to order prioritized diagnostic procedures",
            "Clinical Triage Action: Discharge, Routine Lifestyle Follow-up, Specialist Referral, Emergency ICU Stent Triage",
            "Prescriptive Advice: Pharmacotherapy guidelines (e.g. statins, ACE-inhibitors, anti-platelets)",
        ]
    )
    sensors: List[str] = field(
        default_factory=lambda: [
            "Sphygmomanometer: Resting systolic/diastolic blood pressure (mmHg)",
            "Colorimetric Photometer / Lipid Assay: Serum cholesterol (mg/dL)",
            "Enzymatic Glucometer: Fasting blood glucose (mg/dL)",
            "12-Lead Electrocardiograph (ECG): Peak heart rate and ST-segment depression",
            "Clinical Record Parser: Age, gender, and chest pain symptom reports",
        ]
    )


@dataclass
class AgentAction:
    """Action selected by the clinical agent."""

    triage_tier: str
    risk_score: float  # Percentage [0, 100]
    expected_utility: float
    recommended_pathway: List[str]
    prescriptive_advice: List[str]
    explanation: str


class GoalBasedDiagnosticAgent:
    """
    Goal-Based Agent (Module 1):
    Acts to achieve an explicit clinical goal:
    Goal = { Diagnostic Certainty >= (1 - tau) AND Patient Safety Preserved }.
    """

    def __init__(self, target_entropy: float = 0.20, budget_dollars: float = 2000.0):
        self.target_entropy = target_entropy
        self.budget_dollars = budget_dollars
        self.search_engine = ClinicalPathwaySearch(max_budget_dollars=budget_dollars)

    def formulate_goal(self, current_entropy: float) -> str:
        if current_entropy <= self.target_entropy:
            return "Goal Satisfied: Sufficient diagnostic certainty achieved."
        return f"Active Goal: Reduce diagnostic entropy from {current_entropy:.3f} down to target {self.target_entropy:.3f} within \\${self.budget_dollars:.0f} budget."

    def act(self, prior_risk: float) -> Dict[str, Any]:
        path_result = self.search_engine.astar_search(
            target_entropy=self.target_entropy,
            patient_risk_prior=prior_risk / 100.0,
        )
        return {
            "agent_type": "Goal-Based Agent",
            "target_entropy": self.target_entropy,
            "goal_status": self.formulate_goal(path_result["final_entropy"]),
            "recommended_path": path_result["path"],
            "dollar_cost": path_result["total_dollar_cost"],
            "discomfort": path_result["total_discomfort"],
            "certainty": path_result["certainty_percentage"],
        }


class UtilityBasedDiagnosticAgent:
    """
    Utility-Based Agent (Module 1):
    Maximizes expected multi-attribute clinical utility:
        U(Action) = w_acc * Certainty - w_cost * (Cost / MaxCost) - w_inv * (Invasiveness / MaxInv) - w_risk * MissedRiskPenalty
    Balances diagnostic precision against patient discomfort and financial burden.
    """

    def __init__(
        self,
        w_acc: float = 0.45,
        w_cost: float = 0.20,
        w_inv: float = 0.15,
        w_risk: float = 0.20,
    ):
        self.w_acc = w_acc
        self.w_cost = w_cost
        self.w_inv = w_inv
        self.w_risk = w_risk
        self.peas = PEASSpecification()
        self.fuzzy_engine = MamdaniFIS()
        self.search_engine = ClinicalPathwaySearch()

    def evaluate_utility(
        self,
        certainty_pct: float,
        cost_dollars: float,
        discomfort: float,
        residual_risk: float,
    ) -> float:
        """Computes normalized multi-attribute utility score in [0.0, 1.0]."""
        u_certainty = certainty_pct / 100.0
        u_cost = max(0.0, 1.0 - (cost_dollars / 2500.0))
        u_discomfort = max(0.0, 1.0 - (discomfort / 20.0))
        u_safety = max(0.0, 1.0 - (residual_risk / 100.0))

        utility = (
            self.w_acc * u_certainty
            + self.w_cost * u_cost
            + self.w_inv * u_discomfort
            + self.w_risk * u_safety
        )
        return float(utility)

    def diagnose_and_act(
        self,
        vitals: Dict[str, float],
        mlp_model: Optional[MLPClassifier] = None,
    ) -> AgentAction:
        """
        Synthesizes Fuzzy Inference, ANN predictions, and A* Search
        to generate an optimal clinical action with maximum utility.
        """
        # 1. Soft Computing Fuzzy Inference
        fuzzy_result = self.fuzzy_engine.compute(vitals)
        fuzzy_risk = fuzzy_result["risk_score"]

        # 2. ANN Prediction (if model is supplied)
        ann_risk = None
        if mlp_model is not None:
            # Map input vitals to 5-vector [age, trestbps, chol, fbs, thalach]
            feat_vector = np.array([[
                vitals.get("age", 55.0),
                vitals.get("blood_pressure", 130.0),
                vitals.get("cholesterol", 220.0),
                1.0 if vitals.get("glucose", 90.0) > 120.0 else 0.0,
                vitals.get("heart_rate", 140.0),
            ]])
            ann_proba = float(mlp_model.predict_proba(feat_vector)[0])
            ann_risk = ann_proba * 100.0

        # Blended risk estimate
        composite_risk = (
            0.60 * fuzzy_risk + 0.40 * ann_risk if ann_risk is not None else fuzzy_risk
        )

        # 3. Intelligent Search for Optimal Test Sequencing
        search_res = self.search_engine.astar_search(
            target_entropy=0.18,
            patient_risk_prior=composite_risk / 100.0,
        )

        # 4. Triage and Expected Utility Calculation
        utility = self.evaluate_utility(
            certainty_pct=search_res["certainty_percentage"],
            cost_dollars=search_res["total_dollar_cost"],
            discomfort=search_res["total_discomfort"],
            residual_risk=composite_risk,
        )

        # Action determination
        advice = []
        if composite_risk < 25.0:
            tier = "Tier 1: Discharge & Routine Maintenance"
            advice.append("Maintain regular cardiovascular exercise (150 mins/week moderate aerobic).")
            advice.append("Adhere to Mediterranean dietary pattern low in saturated fats.")
            advice.append("Re-evaluate routine vitals annually.")
        elif composite_risk < 55.0:
            tier = "Tier 2: Outpatient Preventative Cardiology"
            advice.append("Prescribe moderate-intensity HMG-CoA reductase inhibitor (statin) therapy.")
            advice.append("Initiate ambulatory 24-hour blood pressure monitoring.")
            advice.append("Schedule 6-month lipid and fasting glucose panel re-check.")
        elif composite_risk < 80.0:
            tier = "Tier 3: Urgent Cardiology Specialist Referral"
            advice.append("Initiate dual antihypertensive therapy (ACE-inhibitor / ARB + Dihydropyridine CCB).")
            advice.append("Perform urgent Exercise Stress Test or Coronary CT Angiography.")
            advice.append("Administer daily low-dose aspirin (81mg) if no bleeding contraindications.")
        else:
            tier = "Tier 4: Critical Emergency Intervention / Inpatient Stent Triage"
            advice.append("Immediate admission to Coronary Care Unit (CCU).")
            advice.append("Urgent invasive Catheter Coronary Angiogram with view to primary PCI / stent.")
            advice.append("Continuous telemetry ECG monitoring and sublingual nitroglycerin as needed.")

        explanation = (
            f"Fuzzy Risk Engine: {fuzzy_risk:.1f}% ({fuzzy_result['risk_category']}). "
            + (f"Backprop MLP: {ann_risk:.1f}%. " if ann_risk is not None else "")
            + f"Composite Risk: {composite_risk:.1f}%. Optimal pathway requires {len(search_res['path'])} tests "
            + f"at \\${search_res['total_dollar_cost']:.0f} (discomfort index: {search_res['total_discomfort']:.1f})."
        )

        return AgentAction(
            triage_tier=tier,
            risk_score=composite_risk,
            expected_utility=utility,
            recommended_pathway=search_res["path"],
            prescriptive_advice=advice,
            explanation=explanation,
        )


class SoftVsHardComputingBenchmark:
    """
    Demonstrates the fundamental theoretical distinction between
    Hard (Crisp/Binary) Computing and Soft Computing under sensor noise and borderline cases.
    """

    @staticmethod
    def hard_crisp_classifier(bp: float, chol: float, glu: float) -> str:
        """
        Conventional Hard Computing Rule:
        Uses rigid Boolean step functions.
        Fails dramatically near threshold boundaries (e.g. BP 139.9 vs 140.1 mmHg).
        """
        is_hypertensive = bp >= 140.0
        is_hypercholesterolemia = chol >= 240.0
        is_diabetic = glu >= 126.0

        score = int(is_hypertensive) + int(is_hypercholesterolemia) + int(is_diabetic)
        if score == 0:
            return "Low Risk"
        elif score == 1:
            return "Moderate Risk"
        elif score == 2:
            return "High Risk"
        else:
            return "Critical Risk"

    @classmethod
    def simulate_noise_robustness(
        cls,
        base_inputs: Dict[str, float],
        noise_std: float = 4.0,
        trials: int = 40,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Simulates physiological sensor noise (e.g. patient anxiety, cuff movement)
        and measures output variance between Hard Crisp logic and Soft Fuzzy logic.
        """
        rng = np.random.RandomState(random_state)
        fuzzy_fis = MamdaniFIS()

        hard_categories = []
        soft_risk_scores = []

        bp_base = base_inputs.get("blood_pressure", 138.0)
        chol_base = base_inputs.get("cholesterol", 238.0)
        glu_base = base_inputs.get("glucose", 124.0)

        for _ in range(trials):
            # Inject Gaussian sensor noise
            noisy_bp = bp_base + rng.normal(0, noise_std)
            noisy_chol = chol_base + rng.normal(0, noise_std * 1.5)
            noisy_glu = glu_base + rng.normal(0, noise_std)

            # Hard Computing Evaluation
            hard_res = cls.hard_crisp_classifier(noisy_bp, noisy_chol, noisy_glu)
            hard_categories.append(hard_res)

            # Soft Computing Evaluation
            noisy_inputs = {
                "blood_pressure": noisy_bp,
                "cholesterol": noisy_chol,
                "glucose": noisy_glu,
                "heart_rate": base_inputs.get("heart_rate", 140.0),
                "age": base_inputs.get("age", 58.0),
            }
            soft_res = fuzzy_fis.compute(noisy_inputs)
            soft_risk_scores.append(soft_res["risk_score"])

        # Category flipping in hard computing
        unique_hard_cats = len(set(hard_categories))
        soft_std = float(np.std(soft_risk_scores))
        soft_mean = float(np.mean(soft_risk_scores))

        return {
            "trials": trials,
            "noise_std": noise_std,
            "hard_category_distribution": {cat: hard_categories.count(cat) for cat in set(hard_categories)},
            "hard_instability_score": unique_hard_cats,
            "soft_risk_mean": soft_mean,
            "soft_risk_std": soft_std,
            "soft_scores": soft_risk_scores,
            "hard_categories": hard_categories,
        }
