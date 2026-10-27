"""
Mamdani Fuzzy Inference System (FIS) from scratch.
Implements Linguistic Variables, AHA-guided clinical rule base,
Min-Max Mamdani inference, and Centroid Defuzzification.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
from .membership import MembershipFunction, TriangularMF, TrapezoidalMF


class LinguisticVariable:
    """Represents a fuzzy linguistic variable with an associated universe of discourse."""

    def __init__(self, name: str, min_val: float, max_val: float, num_points: int = 500):
        self.name = name
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.num_points = num_points
        self.universe = np.linspace(self.min_val, self.max_val, self.num_points)
        self.terms: Dict[str, MembershipFunction] = {}

    def add_term(self, mf: MembershipFunction):
        self.terms[mf.name] = mf

    def fuzzify(self, crisp_value: float) -> Dict[str, float]:
        clamped_val = np.clip(crisp_value, self.min_val, self.max_val)
        return {term_name: float(mf.evaluate(clamped_val)) for term_name, mf in self.terms.items()}


class FuzzyRule:
    """Represents a single clinical IF-THEN linguistic rule."""

    def __init__(
        self,
        antecedents: Dict[str, str],
        consequent_term: str,
        operator: str = "AND",
        weight: float = 1.0,
        clinical_rationale: str = "",
    ):
        self.antecedents = antecedents
        self.consequent_term = consequent_term
        self.operator = operator.upper()
        self.weight = weight
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

        if self.operator == "AND":
            firing_strength = min(memberships)
        elif self.operator == "OR":
            firing_strength = max(memberships)
        else:
            raise ValueError(f"Unknown rule operator: {self.operator}")

        return firing_strength * self.weight

    def __repr__(self) -> str:
        ant_str = f" {self.operator} ".join([f"{k} IS {v}" for k, v in self.antecedents.items()])
        return f"IF {ant_str} THEN cardiac_risk IS {self.consequent_term}"


class MamdaniFIS:
    """
    Complete Mamdani Fuzzy Inference System for Cardiovascular Risk Stratification.
    Trained and configured based on AHA/ACC Clinical Practice Guidelines.
    """

    def __init__(self):
        self.inputs: Dict[str, LinguisticVariable] = {}
        self.output: LinguisticVariable = None
        self.rules: List[FuzzyRule] = []
        self._setup_default_clinical_system()

    def _setup_default_clinical_system(self):
        # 1. Blood Pressure (trestbps: mmHg)
        bp = LinguisticVariable("blood_pressure", 70, 220)
        bp.add_term(TrapezoidalMF("Low", 70, 70, 85, 100))
        bp.add_term(TriangularMF("Normal", 90, 115, 130))
        bp.add_term(TriangularMF("PreHypertensive", 120, 135, 145))
        bp.add_term(TriangularMF("High", 135, 155, 175))
        bp.add_term(TrapezoidalMF("Crisis", 165, 185, 220, 220))
        self.inputs["blood_pressure"] = bp

        # 2. Total Serum Cholesterol (chol: mg/dL)
        chol = LinguisticVariable("cholesterol", 100, 400)
        chol.add_term(TrapezoidalMF("Desirable", 100, 100, 160, 200))
        chol.add_term(TriangularMF("Borderline", 190, 220, 250))
        chol.add_term(TrapezoidalMF("Dangerous", 235, 270, 400, 400))
        self.inputs["cholesterol"] = chol

        # 3. Fasting Blood Glucose (fbs: mg/dL)
        glu = LinguisticVariable("glucose", 60, 280)
        glu.add_term(TrapezoidalMF("Normal", 60, 60, 85, 105))
        glu.add_term(TriangularMF("PreDiabetic", 95, 115, 130))
        glu.add_term(TrapezoidalMF("Diabetic", 120, 145, 280, 280))
        self.inputs["glucose"] = glu

        # 4. Maximum Heart Rate (thalach: bpm)
        hr = LinguisticVariable("heart_rate", 50, 220)
        hr.add_term(TrapezoidalMF("Bradycardia", 50, 50, 60, 75))
        hr.add_term(TriangularMF("Normal", 70, 135, 175))
        hr.add_term(TrapezoidalMF("Tachycardia", 160, 185, 220, 220))
        self.inputs["heart_rate"] = hr

        # 5. Age (years)
        age = LinguisticVariable("age", 20, 95)
        age.add_term(TrapezoidalMF("Young", 20, 20, 32, 42))
        age.add_term(TriangularMF("MiddleAged", 38, 52, 65))
        age.add_term(TrapezoidalMF("Senior", 60, 72, 95, 95))
        self.inputs["age"] = age

        # Output: Cardiovascular Risk (0% to 100%)
        risk = LinguisticVariable("cardiac_risk", 0, 100, num_points=600)
        risk.add_term(TriangularMF("Low", 0, 12, 28))
        risk.add_term(TriangularMF("Moderate", 22, 42, 62))
        risk.add_term(TriangularMF("High", 55, 72, 88))
        risk.add_term(TrapezoidalMF("Critical", 78, 90, 100, 100))
        self.output = risk

        self._load_clinical_rules()

    def _load_clinical_rules(self):
        self.rules = [
            # Low Risk Profiles
            FuzzyRule(
                {"blood_pressure": "Normal", "cholesterol": "Desirable", "age": "Young"},
                "Low",
                "AND",
                clinical_rationale="Youth with optimal hemodynamic and lipid profile.",
            ),
            FuzzyRule(
                {"blood_pressure": "Normal", "cholesterol": "Desirable", "glucose": "Normal"},
                "Low",
                "AND",
                clinical_rationale="Absence of metabolic and vascular risk factors.",
            ),
            FuzzyRule(
                {"blood_pressure": "Low", "cholesterol": "Desirable", "age": "MiddleAged"},
                "Low",
                "AND",
                clinical_rationale="Controlled blood pressure and lipid balance in middle age.",
            ),
            # Moderate Risk Profiles
            FuzzyRule(
                {"blood_pressure": "PreHypertensive", "cholesterol": "Borderline"},
                "Moderate",
                "AND",
                clinical_rationale="Subclinical elevation of vascular resistance and atherogenic lipids.",
            ),
            FuzzyRule(
                {"age": "MiddleAged", "blood_pressure": "PreHypertensive", "glucose": "PreDiabetic"},
                "Moderate",
                "AND",
                clinical_rationale="Metabolic syndrome onset in middle adulthood.",
            ),
            FuzzyRule(
                {"cholesterol": "Borderline", "glucose": "Normal", "age": "Senior"},
                "Moderate",
                "AND",
                clinical_rationale="Age-related baseline risk with mild lipid abnormality.",
            ),
            FuzzyRule(
                {"heart_rate": "Tachycardia", "blood_pressure": "PreHypertensive"},
                "Moderate",
                "AND",
                clinical_rationale="Elevated resting adrenergic drive and vascular tension.",
            ),
            # High Risk Profiles
            FuzzyRule(
                {"blood_pressure": "High", "cholesterol": "Borderline"},
                "High",
                "AND",
                clinical_rationale="Confirmed stage-1/2 hypertension compounding lipid plaques.",
            ),
            FuzzyRule(
                {"blood_pressure": "High", "glucose": "Diabetic"},
                "High",
                "AND",
                clinical_rationale="Diabetic micro/macrovascular damage exacerbated by hypertension.",
            ),
            FuzzyRule(
                {"cholesterol": "Dangerous", "age": "Senior"},
                "High",
                "AND",
                clinical_rationale="Severe hypercholesterolemia in geriatric patient with stiffened arteries.",
            ),
            FuzzyRule(
                {"age": "Senior", "blood_pressure": "High", "heart_rate": "Bradycardia"},
                "High",
                "AND",
                clinical_rationale="Conduction abnormality coexisting with high afterload.",
            ),
            FuzzyRule(
                {"glucose": "Diabetic", "cholesterol": "Dangerous"},
                "High",
                "AND",
                clinical_rationale="Atherogenic diabetic dyslipidemia accelerates CAD.",
            ),
            # Critical Risk Profiles
            FuzzyRule(
                {"blood_pressure": "Crisis", "cholesterol": "Dangerous"},
                "Critical",
                "OR",
                clinical_rationale="Hypertensive crisis or critical hyperlipidemia carries immediate ischemic threat.",
            ),
            FuzzyRule(
                {"blood_pressure": "High", "cholesterol": "Dangerous", "glucose": "Diabetic"},
                "Critical",
                "AND",
                clinical_rationale="Triple metabolic-hemodynamic pathology; imminent coronary event.",
            ),
            FuzzyRule(
                {"blood_pressure": "Crisis", "age": "Senior"},
                "Critical",
                "AND",
                clinical_rationale="Geriatric hypertensive crisis with extreme risk of stroke or MI.",
            ),
            FuzzyRule(
                {"heart_rate": "Tachycardia", "blood_pressure": "High", "glucose": "Diabetic"},
                "Critical",
                "AND",
                clinical_rationale="Severe myocardial oxygen demand mismatch with vascular disease.",
            ),
        ]

    def fuzzify_all(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        result = {}
        for var_name, crisp_val in inputs.items():
            if var_name in self.inputs:
                result[var_name] = self.inputs[var_name].fuzzify(crisp_val)
        return result

    def compute(self, inputs: Dict[str, float], defuzz_method: str = "centroid") -> Dict[str, Any]:
        """
        Executes complete Mamdani Min-Max Fuzzy Inference pipeline:
        1. Fuzzification
        2. Rule antecedent evaluation (T-norm min / S-norm max)
        3. Mamdani implication (Clipping output membership functions)
        4. Rule aggregation (Maximum over all clipped sets)
        5. Defuzzification (Centroid / MOM / Bisector)
        """
        fuzzified = self.fuzzify_all(inputs)

        # Rule evaluation & implication
        z_universe = self.output.universe
        aggregated_curve = np.zeros_like(z_universe)
        rule_activations = []
        term_activations = {t: 0.0 for t in self.output.terms}

        for rule in self.rules:
            firing_strength = rule.evaluate_antecedent(fuzzified)
            if firing_strength > 1e-4:
                # Consequent MF evaluated on output universe
                consequent_mf = self.output.terms[rule.consequent_term]
                output_mu = consequent_mf.evaluate(z_universe)

                # Mamdani implication: Min clipping
                clipped_mu = np.minimum(firing_strength, output_mu)

                # Aggregation: Max operator
                aggregated_curve = np.maximum(aggregated_curve, clipped_mu)

                term_activations[rule.consequent_term] = max(
                    term_activations[rule.consequent_term], float(firing_strength)
                )

                rule_activations.append({
                    "rule": str(rule),
                    "strength": float(firing_strength),
                    "consequent": rule.consequent_term,
                    "rationale": rule.clinical_rationale,
                })

        # Defuzzification
        total_area = np.sum(aggregated_curve)
        if total_area < 1e-6:
            # Fallback if no rules fired significantly
            crisp_risk = 25.0
        elif defuzz_method.lower() == "centroid":
            # Center of Gravity: sum(z * mu(z)) / sum(mu(z))
            crisp_risk = float(np.sum(z_universe * aggregated_curve) / total_area)
        elif defuzz_method.lower() == "mom":
            # Mean of Maxima
            max_mu = np.max(aggregated_curve)
            crisp_risk = float(np.mean(z_universe[aggregated_curve == max_mu]))
        elif defuzz_method.lower() == "bisector":
            # Bisector: cuts area into two equal halves
            half_area = total_area / 2.0
            cum_area = np.cumsum(aggregated_curve)
            bisect_idx = np.searchsorted(cum_area, half_area)
            crisp_risk = float(z_universe[min(bisect_idx, len(z_universe) - 1)])
        else:
            raise ValueError(f"Unsupported defuzzification method: {defuzz_method}")

        crisp_risk = float(np.clip(crisp_risk, 0.0, 100.0))

        # Risk categorization
        if crisp_risk < 25.0:
            category = "Low Risk"
        elif crisp_risk < 55.0:
            category = "Moderate Risk"
        elif crisp_risk < 80.0:
            category = "High Risk"
        else:
            category = "Critical Risk"

        return {
            "risk_score": crisp_risk,
            "risk_category": category,
            "defuzz_method": defuzz_method,
            "fuzzified_inputs": fuzzified,
            "rule_activations": sorted(rule_activations, key=lambda x: x["strength"], reverse=True),
            "term_activations": term_activations,
            "z_universe": z_universe,
            "aggregated_curve": aggregated_curve,
        }
