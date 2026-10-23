# CSE315 Syllabus Mapping Matrix & Feature Catalog
## Course: Artificial Intelligence and Soft Computing (AISC)

This document provides a comprehensive mapping of all theoretical and algorithmic concepts from the **CSE315: Artificial Intelligence and Soft Computing** syllabus directly to their modular Python implementations, unit tests, and interactive UI visualizer tabs in the **Intelligent Clinical Decision Support System (CDSS)**.

---

### 1. Comprehensive Syllabus-to-Code Mapping Table

| Course Module | Syllabus Topic & Concept | Implementation File | Python Class / Function | UI Dashboard Tab | Unit Test Assertion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Module 1** | Definition of AI & Intelligent Agents | `src/agents/diagnostic_agent.py` | `PEASSpecification` | Sidebar & Tab 1 | `test_peas_specification` |
| **Module 1** | Agents and Environment (Observability, Determinism) | `src/agents/diagnostic_agent.py` | `PEASSpecification.environment_properties` | Tab 1 Architecture View | `test_peas_specification` |
| **Module 1** | Goal-Based Agent Architecture | `src/agents/diagnostic_agent.py` | `GoalBasedDiagnosticAgent` | Tab 1 & Tab 3 | `test_goal_based_agent` |
| **Module 1** | Utility-Based Agent Architecture | `src/agents/diagnostic_agent.py` | `UtilityBasedDiagnosticAgent` | Tab 1 (Expected Utility Meter) | `test_utility_based_agent` |
| **Module 1** | Soft vs. Hard Computing Foundations | `src/agents/diagnostic_agent.py` | `SoftVsHardComputingBenchmark` | Tab 1 (Noise Robustness Simulator) | `test_soft_vs_hard_computing_noise_stability` |
| **Module 1** | Handling Uncertainty, Imprecision & Noise | `src/agents/diagnostic_agent.py` | `simulate_noise_robustness()` | Tab 1 Interactive Experiment | `test_soft_vs_hard_computing_noise_stability` |
| **Module 2** | Informed (Heuristic) Search: A\* Algorithm | `src/search/astar_pathway.py` | `ClinicalPathwaySearch.astar_search` | Tab 3 (A\* Pathway Optimizer) | `test_astar_search_optimality` |
| **Module 2** | Heuristic Admissibility & Consistency ($h(n) \le h^*(n)$) | `src/search/astar_pathway.py` | `admissible_heuristic()` | Tab 3 Search Trace | `test_heuristic_admissibility` |
| **Module 2** | Diagnostic Shannon Entropy Reduction | `src/search/astar_pathway.py` | `ClinicalTest.entropy_reduction` | Tab 3 Entropy Slider & Trace | `test_astar_search_optimality` |
| **Module 2** | Greedy Best-First Search Comparison | `src/search/astar_pathway.py` | `greedy_best_first_search()` | Tab 3 Multi-Algorithm Cards | `test_search_comparison` |
| **Module 2** | Steepest-Ascent Hill Climbing Search | `src/search/astar_pathway.py` | `hill_climbing_search()` | Tab 3 Multi-Algorithm Cards | `test_search_comparison` |
| **Module 3** | Multi-Layer Feedforward ANN Architecture | `src/models/mlp_classifier.py` | `MLPClassifier` | Tab 1 (MLP Disease Proba) | `test_mlp_forward_and_backward` |
| **Module 3** | Supervised Backpropagation Algorithm | `src/models/mlp_classifier.py` | `MLPClassifier.train_step()` | Tab 1 Metric Card | `test_mlp_forward_and_backward` |
| **Module 3** | Generalized Delta Learning Rule | `src/models/mlp_classifier.py` | `delta3`, `delta2`, `delta1` | Tab 1 & Tab 4 | `test_mlp_forward_and_backward` |
| **Module 3** | Momentum & L2 Weight Regularization | `src/models/mlp_classifier.py` | `vW1`, `vW2`, `vW3` updates | Background Cache | `test_mlp_forward_and_backward` |
| **Module 3** | ANN Evaluation Metrics (Confusion Matrix, F1, AUC) | `src/models/mlp_classifier.py` | `MLPClassifier.evaluate()` | Sidebar Status | `test_train_mlp_on_dataset` |
| **Module 4** | Fuzzy Set Theory vs. Crisp Sets | `src/fuzzy/membership.py` | `MembershipFunction` | Tab 2 MF Curve Visualizer | `test_membership_functions` |
| **Module 4** | Triangular Membership Function ($\text{trimf}$) | `src/fuzzy/membership.py` | `TriangularMF` | Tab 2 Curve Visualizer | `test_membership_functions` |
| **Module 4** | Trapezoidal Membership Function ($\text{trapmf}$) | `src/fuzzy/membership.py` | `TrapezoidalMF` | Tab 2 Curve Visualizer | `test_membership_functions` |
| **Module 4** | Gaussian Membership Function ($\text{gaussmf}$) | `src/fuzzy/membership.py` | `GaussianMF` | Core Library | `test_membership_functions` |
| **Module 4** | Mamdani Linguistic IF-THEN Clinical Rules | `src/fuzzy/mamdani_fis.py` | `FuzzyRule`, `_load_clinical_rules` | Tab 1 (Fired Rules Table) | `test_mamdani_inference_and_defuzzification` |
| **Module 4** | Fuzzy Inference: Min T-norm & Max S-norm | `src/fuzzy/mamdani_fis.py` | `FuzzyRule.evaluate_antecedent` | Tab 1 & Tab 2 | `test_mamdani_inference_and_defuzzification` |
| **Module 4** | Mamdani Minimum Implication & Aggregation | `src/fuzzy/mamdani_fis.py` | `MamdaniFIS.compute()` | Tab 2 Aggregation Shaded Plot | `test_mamdani_inference_and_defuzzification` |
| **Module 4** | Centroid (Center of Gravity) Defuzzification | `src/fuzzy/mamdani_fis.py` | `compute(defuzz_method='centroid')`| Tab 1 Meter & Tab 2 Plot | `test_mamdani_inference_and_defuzzification` |
| **Module 4** | Alternative Defuzzification (MOM, Bisector) | `src/fuzzy/mamdani_fis.py` | `defuzz_method='mom'/'bisector'` | Tab 2 Method Comparison | `test_mamdani_inference_and_defuzzification` |
| **Module 4** | Takagi-Sugeno-Kang (TSK) FIS System | `src/fuzzy/sugeno_fis.py` | `SugenoFIS`, `SugenoRule` | Tab 2 Sugeno Comparison Card | `test_sugeno_inference` |
| **Module 4** | ANFIS 5-Layer Theoretical Architecture | `src/fuzzy/sugeno_fis.py` | `anfis_architecture_breakdown()` | Tab 2 ANFIS Breakdown Table | `test_sugeno_inference` |
| **Module 5** | Binary Genetic Algorithm (GA) Representation | `src/optimization/genetic_selector.py`| `GeneticFeatureSelector` | Tab 4 (GA Tab) | `test_genetic_algorithm_feature_selector` |
| **Module 5** | Multi-Objective Fitness (Accuracy + Parsimony) | `src/optimization/genetic_selector.py`| `fitness_function()` | Tab 4 Convergence Plot | `test_genetic_algorithm_feature_selector` |
| **Module 5** | GA Operators: Roulette-Wheel & Tournament Selection | `src/optimization/genetic_selector.py`| `roulette_wheel_selection()`, `tournament_selection()` | Tab 4 Config | `test_genetic_algorithm_feature_selector` |
| **Module 5** | GA Operators: Single-Point Crossover & Mutation | `src/optimization/genetic_selector.py`| `single_point_crossover()`, `bit_flip_mutation()` | Tab 4 Config | `test_genetic_algorithm_feature_selector` |
| **Module 5** | Holland's Schema Theorem Formalization | `src/optimization/genetic_selector.py`| `schema_theorem_summary()` | Tab 4 Schema Theorem Callout | `test_genetic_algorithm_feature_selector` |
| **Module 5** | Particle Swarm Optimization (PSO) Formulation | `src/optimization/pso_tuner.py` | `ParticleSwarmTuner`, `Particle` | Tab 4 (PSO Tab) | `test_pso_tuner` |
| **Module 5** | PSO Velocity & Position Updates ($p\text{Best}$, $g\text{Best}$) | `src/optimization/pso_tuner.py` | `fit()`, cognitive & social terms | Tab 4 Convergence Plot | `test_pso_tuner` |
| **Module 5** | Linear Inertia Weight ($w$) Decay Strategy | `src/optimization/pso_tuner.py` | `w = w_max - t * delta_w` | Tab 4 Convergence Log | `test_pso_tuner` |
| **Module 5** | Neuro-Fuzzy / Metaheuristic Membership Tuning | `src/optimization/pso_tuner.py` | `_apply_params_to_fis()` | Tab 4 Tuned Params JSON | `test_pso_tuner` |

---

### 2. Architectural Modularity Summary

- **Pure NumPy Implementation:** All neural network gradient backpropagation, matrix operations, fuzzy membership interpolations, defuzzification integrations, and search priority queues are implemented cleanly from first principles without blackbox fuzzy/deep learning libraries.
- **Offline & Standalone Operation:** The dataset is fully localized in `data/heart.csv` (303 records, calibrated UCI Cleveland distribution). The application does not require internet access, third-party cloud APIs, or external databases.
- **Verification Guarantee:** Every single class is validated by the unit test suite in `tests/test_cdss.py` (`14 passed tests in 0.6s`).
