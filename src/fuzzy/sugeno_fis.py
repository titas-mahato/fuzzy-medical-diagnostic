"""
Sugeno Fuzzy Inference System (Takagi-Sugeno-Kang) & ANFIS Architecture.
"""

from typing import Dict, List, Any
import numpy as np
from .membership import TriangularMF, TrapezoidalMF


class SugenoRule:
    """
    Takagi-Sugeno-Kang Rule:
    IF x1 is A1 AND x2 is A2 ... THEN z = p0 + p1*x1 + p2*x2 + ...
    For Zero-order Sugeno: z = constant (p0).
    For First-order Sugeno: z = linear function of inputs.
    """

    def __init__(
        self,
        antecedents: Dict[str, str],
        coefficients: Dict[str, float],
        intercept: float,
        operator: str = "AND",
        clinical_rationale: str = "",
    ):
        self.antecedents = antecedents
        self.coefficients = coefficients  # e.g., {'blood_pressure': 0.2, 'cholesterol': 0.1}
        self.intercept = float(intercept)
        self.operator = operator.upper()
        self.clinical_rationale = clinical_rationale

    def evaluate_antecedent(self, fuzzified_inputs: Dict[str, Dict[str, float]]) -> float:
        memberships = []
        for var_name, term_name in self.antecedents.items():
            if var_name in fuzzified_inputs and term_name in fuzzified_inputs[var_name]:
                memberships.append(fuzzified_inputs[var_name][term_name])
            else:
                memberships.append(0.0)

        if not memberships:
            return 0.0

        return min(memberships) if self.operator == "AND" else max(memberships)

    def evaluate_consequent(self, crisp_inputs: Dict[str, float]) -> float:
        output_z = self.intercept
        for var_name, coef in self.coefficients.items():
            val = crisp_inputs.get(var_name, 0.0)
            output_z += coef * val
        return output_z


class SugenoFIS:
    """
    Sugeno (TSK) Fuzzy Inference System.
    Computes output via Weighted Average Defuzzification:
        z* = (sum_{i=1}^R w_i * z_i) / (sum_{i=1}^R w_i)
    """

    def __init__(self, order: int = 1):
        self.order = order
        self.rules: List[SugenoRule] = []
        self._setup_default_sugeno_system()

    def _setup_default_sugeno_system(self):
        # Sample clinical rules mapped to linear or constant functions
        if self.order == 0:
            # Zero-order constants: Low=15%, Moderate=45%, High=75%, Critical=95%
            self.rules = [
                SugenoRule({"blood_pressure": "Normal", "cholesterol": "Desirable"}, {}, 15.0, "AND", "Low baseline"),
                SugenoRule({"blood_pressure": "PreHypertensive", "cholesterol": "Borderline"}, {}, 45.0, "AND", "Moderate onset"),
                SugenoRule({"blood_pressure": "High", "cholesterol": "Borderline"}, {}, 72.0, "AND", "High vascular risk"),
                SugenoRule({"blood_pressure": "Crisis", "cholesterol": "Dangerous"}, {}, 94.0, "OR", "Critical emergency"),
            ]
        else:
            # First-order polynomials: z = p0 + p1*BP + p2*Chol + p3*Age
            self.rules = [
                SugenoRule(
                    {"blood_pressure": "Normal", "cholesterol": "Desirable"},
                    {"blood_pressure": 0.05, "cholesterol": 0.03, "age": 0.1},
                    5.0,
                    "AND",
                    "Low risk linear response",
                ),
                SugenoRule(
                    {"blood_pressure": "PreHypertensive", "cholesterol": "Borderline"},
                    {"blood_pressure": 0.12, "cholesterol": 0.08, "age": 0.2},
                    15.0,
                    "AND",
                    "Moderate risk linear response",
                ),
                SugenoRule(
                    {"blood_pressure": "High", "cholesterol": "Borderline"},
                    {"blood_pressure": 0.22, "cholesterol": 0.12, "age": 0.3},
                    25.0,
                    "AND",
                    "High risk linear response",
                ),
                SugenoRule(
                    {"blood_pressure": "Crisis", "cholesterol": "Dangerous"},
                    {"blood_pressure": 0.30, "cholesterol": 0.18, "age": 0.35},
                    40.0,
                    "OR",
                    "Critical risk linear response",
                ),
            ]

    def compute(self, crisp_inputs: Dict[str, float], fuzzified_inputs: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        weights = []
        consequents = []
        rule_details = []

        for rule in self.rules:
            w = rule.evaluate_antecedent(fuzzified_inputs)
            z = rule.evaluate_consequent(crisp_inputs)
            weights.append(w)
            consequents.append(z)
            rule_details.append({
                "weight": float(w),
                "z": float(z),
                "rationale": rule.clinical_rationale,
            })

        weights_arr = np.array(weights)
        consequents_arr = np.array(consequents)
        sum_w = np.sum(weights_arr)

        if sum_w < 1e-6:
            risk = 30.0
        else:
            risk = float(np.sum(weights_arr * consequents_arr) / sum_w)

        risk = float(np.clip(risk, 0.0, 100.0))

        return {
            "risk_score": risk,
            "order": self.order,
            "rule_weights": weights,
            "rule_consequents": consequents,
            "rule_details": rule_details,
        }

    @staticmethod
    def anfis_architecture_breakdown() -> Dict[str, str]:
        """Provides theoretical 5-layer breakdown of ANFIS for viva presentation."""
        return {
            "Layer 1 (Fuzzification Nodes)": "Generates membership grades mu_A_i(x) using bell or triangular shapes with tunable premise parameters {a, b, c}.",
            "Layer 2 (Rule Nodes / Firing Strength)": "Computes T-norm (product / min) to obtain firing strength w_i = mu_A_i(x) * mu_B_i(y).",
            "Layer 3 (Normalization Nodes)": "Normalizes firing strengths w_bar_i = w_i / sum(w_k).",
            "Layer 4 (Defuzzification / Consequent Nodes)": "Computes polynomial output f_i = w_bar_i * (p_i*x + q_i*y + r_i) with tunable consequent parameters.",
            "Layer 5 (Summation Node)": "Computes overall crisp network output: y = sum(w_bar_i * f_i).",
        }
