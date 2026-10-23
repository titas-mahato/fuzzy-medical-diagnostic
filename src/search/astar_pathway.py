"""
Informed A* Search and Clinical Pathway Optimization.
Models clinical diagnostic test sequencing as a state space graph search
balancing financial cost, patient discomfort, and diagnostic uncertainty reduction.
Course: CSE315 - Module 2: Problem Solving by Intelligent Search
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import heapq
import numpy as np


class ClinicalTest:
    """Represents an actionable clinical diagnostic investigation."""

    def __init__(
        self,
        name: str,
        cost_dollars: float,
        discomfort_score: float,  # Scale 1-10 (1=Non-invasive, 10=Surgical/Severe)
        entropy_reduction: float,  # Information gain in diagnostic certainty (0.0 to 1.0)
        description: str = "",
    ):
        self.name = name
        self.cost_dollars = float(cost_dollars)
        self.discomfort_score = float(discomfort_score)
        self.entropy_reduction = float(entropy_reduction)
        self.description = description

    def total_step_cost(self, discomfort_weight: float = 12.0) -> float:
        """Composite clinical cost combining financial burden and patient invasiveness."""
        return self.cost_dollars + (discomfort_weight * self.discomfort_score)


class SearchNode:
    """Node in the clinical test pathway search tree."""

    def __init__(
        self,
        tests_performed: Tuple[str, ...],
        current_entropy: float,
        g_cost: float,
        h_cost: float,
        parent: Optional["SearchNode"] = None,
        last_test: Optional[str] = None,
    ):
        self.tests_performed = tests_performed
        self.current_entropy = current_entropy
        self.g = g_cost  # Path cost so far: g(n)
        self.h = h_cost  # Heuristic estimate to goal: h(n)
        self.f = g_cost + h_cost  # Evaluation function: f(n) = g(n) + h(n)
        self.parent = parent
        self.last_test = last_test

    def __lt__(self, other: "SearchNode") -> bool:
        return self.f < other.f


class ClinicalPathwaySearch:
    """
    Search engine comparing Informed A* Search, Greedy Best-First,
    and Steepest-Ascent Hill Climbing for diagnostic test selection.
    """

    def __init__(self, discomfort_weight: float = 12.0, max_budget_dollars: float = 2500.0):
        self.discomfort_weight = discomfort_weight
        self.max_budget = max_budget_dollars
        self.test_catalog: Dict[str, ClinicalTest] = self._init_test_catalog()
        self.max_efficiency = self._compute_max_efficiency()

    def _init_test_catalog(self) -> Dict[str, ClinicalTest]:
        return {
            "Basic Vitals & History": ClinicalTest(
                "Basic Vitals & History",
                cost_dollars=15.0,
                discomfort_score=1.0,
                entropy_reduction=0.15,
                description="Sphygmomanometer, pulse, BMI, and clinical anamnesis.",
            ),
            "Fasting Lipid & Glucose Panel": ClinicalTest(
                "Fasting Lipid & Glucose Panel",
                cost_dollars=45.0,
                discomfort_score=2.0,
                entropy_reduction=0.25,
                description="Venipuncture blood test for total cholesterol, HDL/LDL, and glucose.",
            ),
            "12-Lead Electrocardiogram (ECG)": ClinicalTest(
                "12-Lead Electrocardiogram (ECG)",
                cost_dollars=75.0,
                discomfort_score=2.0,
                entropy_reduction=0.30,
                description="Electrophysiological recording for arrhythmias, ST-elevation/depression.",
            ),
            "Echocardiogram (Echo)": ClinicalTest(
                "Echocardiogram (Echo)",
                cost_dollars=350.0,
                discomfort_score=3.0,
                entropy_reduction=0.55,
                description="Transthoracic ultrasound assessing ejection fraction and valvular structure.",
            ),
            "Treadmill Exercise Stress Test": ClinicalTest(
                "Treadmill Exercise Stress Test",
                cost_dollars=400.0,
                discomfort_score=6.0,
                entropy_reduction=0.65,
                description="Provocative exercise test for inducible myocardial ischemia.",
            ),
            "Coronary CT Angiography (CCTA)": ClinicalTest(
                "Coronary CT Angiography (CCTA)",
                cost_dollars=900.0,
                discomfort_score=4.5,
                entropy_reduction=0.85,
                description="Non-invasive 3D iodine contrast imaging of coronary arterial lumen.",
            ),
            "Catheter Coronary Angiogram": ClinicalTest(
                "Catheter Coronary Angiogram",
                cost_dollars=2200.0,
                discomfort_score=8.5,
                entropy_reduction=0.98,
                description="Invasive fluoroscopic catheterization (Gold Standard definitive diagnosis).",
            ),
        }

    def _compute_max_efficiency(self) -> float:
        """Maximum entropy reduction per cost unit across all available tests."""
        efficiencies = [
            t.entropy_reduction / t.total_step_cost(self.discomfort_weight)
            for t in self.test_catalog.values()
        ]
        return max(efficiencies)

    def admissible_heuristic(self, remaining_entropy: float, target_entropy: float) -> float:
        """
        Admissible & Consistent Heuristic h(n):
        Estimates the minimum conceivable cost to bridge remaining diagnostic entropy:
            h(n) = max(0, (H - H_target) / max_efficiency)
        Since max_efficiency >= (delta_H / step_cost) for every test, h(n) never overestimates
        the actual remaining cost h*(n), guaranteeing A* finds the globally optimal pathway.
        """
        deficit = max(0.0, remaining_entropy - target_entropy)
        return deficit / self.max_efficiency

    def astar_search(
        self,
        target_entropy: float = 0.20,
        patient_risk_prior: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Informed A* Search:
        Expands nodes minimizing f(n) = g(n) + h(n).
        """
        # Initial entropy of patient state: Shannon entropy H(p) = -p log2(p) - (1-p) log2(1-p)
        eps = 1e-6
        p = np.clip(patient_risk_prior, eps, 1.0 - eps)
        initial_entropy = float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))

        start_h = self.admissible_heuristic(initial_entropy, target_entropy)
        start_node = SearchNode(tests_performed=(), current_entropy=initial_entropy, g_cost=0.0, h_cost=start_h)

        open_set: List[SearchNode] = []
        heapq.heappush(open_set, start_node)
        closed_set: Set[Tuple[str, ...]] = set()

        nodes_expanded = 0
        search_trace = []

        while open_set:
            current = heapq.heappop(open_set)
            nodes_expanded += 1

            search_trace.append({
                "step": nodes_expanded,
                "tests": list(current.tests_performed),
                "entropy": round(current.current_entropy, 4),
                "g": round(current.g, 2),
                "h": round(current.h, 2),
                "f": round(current.f, 2),
            })

            # Goal Test: Entropy reduced to or below clinical threshold
            if current.current_entropy <= target_entropy:
                return self._build_result(current, nodes_expanded, search_trace, "A* Search (Optimal)")

            if current.tests_performed in closed_set:
                continue
            closed_set.add(current.tests_performed)

            # Expand successors
            for test_name, test in self.test_catalog.items():
                if test_name not in current.tests_performed:
                    new_tests = tuple(sorted(list(current.tests_performed) + [test_name]))
                    if new_tests in closed_set:
                        continue

                    step_cost = test.total_step_cost(self.discomfort_weight)
                    new_g = current.g + step_cost

                    # Check financial constraint
                    total_dollar_cost = sum(self.test_catalog[t].cost_dollars for t in new_tests)
                    if total_dollar_cost > self.max_budget:
                        continue

                    # Reduced diagnostic entropy
                    new_entropy = max(0.0, current.current_entropy * (1.0 - test.entropy_reduction))
                    new_h = self.admissible_heuristic(new_entropy, target_entropy)

                    child_node = SearchNode(
                        tests_performed=new_tests,
                        current_entropy=new_entropy,
                        g_cost=new_g,
                        h_cost=new_h,
                        parent=current,
                        last_test=test_name,
                    )
                    heapq.heappush(open_set, child_node)

        # Fallback if target entropy unachievable within budget
        return self._build_result(current, nodes_expanded, search_trace, "A* (Budget Exceeded)")

    def greedy_best_first_search(
        self,
        target_entropy: float = 0.20,
        patient_risk_prior: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Greedy Best-First Search:
        Evaluates nodes solely based on heuristic: f(n) = h(n).
        Sacrifices cost optimality for speed; susceptible to high cost tests with large entropy drops.
        """
        eps = 1e-6
        p = np.clip(patient_risk_prior, eps, 1.0 - eps)
        initial_entropy = float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))

        start_h = self.admissible_heuristic(initial_entropy, target_entropy)
        start_node = SearchNode(tests_performed=(), current_entropy=initial_entropy, g_cost=0.0, h_cost=start_h)
        start_node.f = start_h  # Only heuristic

        open_set: List[SearchNode] = []
        heapq.heappush(open_set, start_node)
        closed_set: Set[Tuple[str, ...]] = set()

        nodes_expanded = 0
        search_trace = []

        while open_set:
            current = heapq.heappop(open_set)
            nodes_expanded += 1

            search_trace.append({
                "step": nodes_expanded,
                "tests": list(current.tests_performed),
                "entropy": round(current.current_entropy, 4),
                "g": round(current.g, 2),
                "h": round(current.h, 2),
                "f": round(current.f, 2),
            })

            if current.current_entropy <= target_entropy:
                return self._build_result(current, nodes_expanded, search_trace, "Greedy Best-First")

            if current.tests_performed in closed_set:
                continue
            closed_set.add(current.tests_performed)

            for test_name, test in self.test_catalog.items():
                if test_name not in current.tests_performed:
                    new_tests = tuple(sorted(list(current.tests_performed) + [test_name]))
                    if new_tests in closed_set:
                        continue

                    step_cost = test.total_step_cost(self.discomfort_weight)
                    new_g = current.g + step_cost
                    new_entropy = max(0.0, current.current_entropy * (1.0 - test.entropy_reduction))
                    new_h = self.admissible_heuristic(new_entropy, target_entropy)

                    child_node = SearchNode(
                        tests_performed=new_tests,
                        current_entropy=new_entropy,
                        g_cost=new_g,
                        h_cost=new_h,
                        parent=current,
                        last_test=test_name,
                    )
                    child_node.f = new_h  # Greedy evaluation
                    heapq.heappush(open_set, child_node)

        return self._build_result(current, nodes_expanded, search_trace, "Greedy (Budget Exceeded)")

    def hill_climbing_search(
        self,
        target_entropy: float = 0.20,
        patient_risk_prior: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Steepest-Ascent Hill Climbing:
        Local search moving strictly to the single best immediate neighbor.
        Prone to getting stuck in local optima or plateaus.
        """
        eps = 1e-6
        p = np.clip(patient_risk_prior, eps, 1.0 - eps)
        current_entropy = float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))

        current_tests: Tuple[str, ...] = ()
        cumulative_g = 0.0
        nodes_expanded = 0
        search_trace = []

        while current_entropy > target_entropy:
            nodes_expanded += 1
            best_neighbor = None
            best_entropy = current_entropy
            best_test = None
            best_step_cost = 0.0

            # Evaluate all immediate single-test additions
            for test_name, test in self.test_catalog.items():
                if test_name not in current_tests:
                    cand_entropy = max(0.0, current_entropy * (1.0 - test.entropy_reduction))
                    if cand_entropy < best_entropy:
                        best_entropy = cand_entropy
                        best_test = test_name
                        best_step_cost = test.total_step_cost(self.discomfort_weight)

            search_trace.append({
                "step": nodes_expanded,
                "tests": list(current_tests),
                "entropy": round(current_entropy, 4),
                "g": round(cumulative_g, 2),
                "h": round(best_entropy, 4),
                "f": round(cumulative_g + best_entropy, 2),
            })

            # Check if progress was made (steepest descent on entropy)
            if best_test is None or best_entropy >= current_entropy:
                # Trapped in local optimum
                break

            current_tests = tuple(sorted(list(current_tests) + [best_test]))
            current_entropy = best_entropy
            cumulative_g += best_step_cost

        final_node = SearchNode(
            tests_performed=current_tests,
            current_entropy=current_entropy,
            g_cost=cumulative_g,
            h_cost=0.0,
        )
        return self._build_result(final_node, nodes_expanded, search_trace, "Steepest-Ascent Hill Climbing")

    def _build_result(
        self,
        node: SearchNode,
        nodes_expanded: int,
        trace: List[Dict[str, Any]],
        algorithm: str,
    ) -> Dict[str, Any]:
        path_tests = list(node.tests_performed)
        dollar_cost = sum(self.test_catalog[t].cost_dollars for t in path_tests)
        discomfort = sum(self.test_catalog[t].discomfort_score for t in path_tests)

        return {
            "algorithm": algorithm,
            "path": path_tests,
            "final_entropy": float(node.current_entropy),
            "certainty_percentage": float(np.clip((1.0 - node.current_entropy) * 100.0, 0.0, 100.0)),
            "total_dollar_cost": float(dollar_cost),
            "total_discomfort": float(discomfort),
            "total_evaluation_f": float(node.f),
            "nodes_expanded": nodes_expanded,
            "search_trace": trace,
        }

    def compare_search_algorithms(self, target_entropy: float = 0.20, prior: float = 0.50) -> List[Dict[str, Any]]:
        """Runs A*, Greedy, and Hill Climbing side-by-side for comparative study."""
        astar_res = self.astar_search(target_entropy, prior)
        greedy_res = self.greedy_best_first_search(target_entropy, prior)
        hc_res = self.hill_climbing_search(target_entropy, prior)
        return [astar_res, greedy_res, hc_res]
