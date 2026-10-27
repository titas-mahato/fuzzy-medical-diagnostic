"""
Unit and Integration Test Suite for CDSS.
Validates all core system components:
- Diagnostic Agents (PEAS, Goal & Utility, Soft vs Hard)
- Intelligent Search (A*, Greedy, Hill Climbing, Admissibility)
- Artificial Neural Networks (MLP Backpropagation, Delta Rule, Metrics)
- Fuzzy Logic (Membership, Mamdani FIS, Centroid Defuzzification, Sugeno FIS)
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.fuzzy.membership import TriangularMF, TrapezoidalMF, GaussianMF
from src.fuzzy.mamdani_fis import MamdaniFIS, LinguisticVariable
from src.fuzzy.sugeno_fis import SugenoFIS
from src.models.mlp_classifier import MLPClassifier, train_mlp_on_dataset
from src.search.astar_pathway import ClinicalPathwaySearch, ClinicalTest
from src.agents.diagnostic_agent import (
    PEASSpecification,
    GoalBasedDiagnosticAgent,
    UtilityBasedDiagnosticAgent,
    SoftVsHardComputingBenchmark,
)


class TestAgentsAndSoftComputing(unittest.TestCase):
    """AI & Soft Computing Foundations Tests."""

    def test_peas_specification(self):
        peas = PEASSpecification()
        self.assertTrue(len(peas.performance_measures) >= 3)
        self.assertIn("Observability", peas.environment_properties)
        self.assertTrue(len(peas.actuators) >= 3)
        self.assertTrue(len(peas.sensors) >= 4)

    def test_goal_based_agent(self):
        agent = GoalBasedDiagnosticAgent(target_entropy=0.20, budget_dollars=1500.0)
        res = agent.act(prior_risk=65.0)
        self.assertIn("recommended_path", res)
        self.assertLessEqual(res["dollar_cost"], 1500.0)
        self.assertGreater(res["certainty"], 50.0)

    def test_utility_based_agent(self):
        agent = UtilityBasedDiagnosticAgent()
        vitals = {
            "blood_pressure": 150.0,
            "cholesterol": 260.0,
            "glucose": 130.0,
            "heart_rate": 155.0,
            "age": 62.0,
        }
        action = agent.diagnose_and_act(vitals)
        self.assertIn("Tier", action.triage_tier)
        self.assertTrue(0.0 <= action.risk_score <= 100.0)
        self.assertTrue(0.0 <= action.expected_utility <= 1.0)
        self.assertTrue(len(action.prescriptive_advice) > 0)

    def test_soft_vs_hard_computing_noise_stability(self):
        benchmark = SoftVsHardComputingBenchmark.simulate_noise_robustness(
            {"blood_pressure": 139.0, "cholesterol": 239.0, "glucose": 125.0},
            noise_std=3.0,
            trials=30,
        )
        self.assertGreaterEqual(benchmark["hard_instability_score"], 2)
        self.assertGreater(benchmark["soft_risk_mean"], 0.0)


class TestIntelligentSearch(unittest.TestCase):
    """Problem Solving by Intelligent Search Tests."""

    def setUp(self):
        self.searcher = ClinicalPathwaySearch(discomfort_weight=10.0, max_budget_dollars=2000.0)

    def test_heuristic_admissibility(self):
        # Heuristic must never overestimate actual cost
        for t_name, test in self.searcher.test_catalog.items():
            eff = test.entropy_reduction / test.total_step_cost(10.0)
            self.assertLessEqual(eff, self.searcher.max_efficiency + 1e-9)

        h_val = self.searcher.admissible_heuristic(0.8, 0.2)
        self.assertGreaterEqual(h_val, 0.0)

    def test_astar_search_optimality(self):
        res = self.searcher.astar_search(target_entropy=0.20, patient_risk_prior=0.60)
        self.assertIn("path", res)
        self.assertLessEqual(res["final_entropy"], 0.20)
        self.assertLessEqual(res["total_dollar_cost"], 2000.0)

    def test_search_comparison(self):
        comparisons = self.searcher.compare_search_algorithms(target_entropy=0.25, prior=0.55)
        self.assertEqual(len(comparisons), 3)
        alg_names = [c["algorithm"] for c in comparisons]
        self.assertTrue(any("A*" in name for name in alg_names))
        self.assertTrue(any("Greedy" in name for name in alg_names))
        self.assertTrue(any("Hill" in name for name in alg_names))


class TestNeuralNetworks(unittest.TestCase):
    """Artificial Neural Networks Tests."""

    def test_mlp_forward_and_backward(self):
        mlp = MLPClassifier(input_dim=5, hidden_dims=(8, 4), learning_rate=0.05)
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, size=20)

        mlp.fit(X, y, epochs=10, batch_size=10)
        self.assertEqual(len(mlp.training_history), 10)

        preds = mlp.predict(X)
        self.assertEqual(len(preds), 20)
        proba = mlp.predict_proba(X)
        self.assertTrue(np.all((proba >= 0.0) & (proba <= 1.0)))

        metrics = mlp.evaluate(X, y)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("confusion_matrix", metrics)

    def test_train_mlp_on_dataset(self):
        csv_path = "D:/Zenith/Projects/fuzzy-medical-diagnostic/data/heart.csv"
        mlp, metrics, df = train_mlp_on_dataset(csv_path, epochs=30)
        self.assertGreater(metrics["accuracy"], 0.60)
        self.assertGreater(metrics["roc_auc"], 0.60)


class TestFuzzyLogic(unittest.TestCase):
    """Fuzzy Logic Inference Systems Tests."""

    def test_membership_functions(self):
        tri = TriangularMF("Normal", 10, 20, 30)
        self.assertEqual(tri(10), 0.0)
        self.assertEqual(tri(20), 1.0)
        self.assertEqual(tri(25), 0.5)
        self.assertEqual(tri(35), 0.0)

        trap = TrapezoidalMF("Dangerous", 20, 30, 40, 50)
        self.assertEqual(trap(15), 0.0)
        self.assertEqual(trap(35), 1.0)
        self.assertEqual(trap(45), 0.5)
        self.assertEqual(trap(55), 0.0)

        gauss = GaussianMF("Smooth", mean=50, sigma=10)
        self.assertAlmostEqual(gauss(50), 1.0)
        self.assertAlmostEqual(gauss(60), np.exp(-0.5))

    def test_mamdani_inference_and_defuzzification(self):
        fis = MamdaniFIS()
        inputs = {
            "blood_pressure": 165.0,  # High
            "cholesterol": 280.0,     # Dangerous
            "glucose": 140.0,         # Diabetic
            "heart_rate": 160.0,      # Tachycardia
            "age": 68.0,              # Senior
        }
        res = fis.compute(inputs, defuzz_method="centroid")
        self.assertTrue(0.0 <= res["risk_score"] <= 100.0)
        self.assertIn(res["risk_category"], ["High Risk", "Critical Risk"])
        self.assertTrue(len(res["rule_activations"]) > 0)
        self.assertIsNotNone(res["aggregated_curve"])

    def test_sugeno_inference(self):
        fis = MamdaniFIS()
        fuzzified = fis.fuzzify_all({"blood_pressure": 150.0, "cholesterol": 230.0, "age": 55.0})
        sugeno = SugenoFIS(order=1)
        res = sugeno.compute(
            {"blood_pressure": 150.0, "cholesterol": 230.0, "age": 55.0},
            fuzzified,
        )
        self.assertTrue(0.0 <= res["risk_score"] <= 100.0)
        breakdown = SugenoFIS.anfis_architecture_breakdown()
        self.assertIn("Layer 1 (Fuzzification Nodes)", breakdown)
        self.assertIn("membership grades", breakdown["Layer 1 (Fuzzification Nodes)"])


if __name__ == "__main__":
    unittest.main()
