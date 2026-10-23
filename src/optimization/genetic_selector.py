"""
Genetic Algorithm (GA) for Clinical Biomarker Feature Selection.
Optimizes parsimonious diagnostic panels balancing accuracy and testing cost.
Course: CSE315 - Module 5: Evolutionary Algorithms
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


class GeneticFeatureSelector:
    """
    Binary Genetic Algorithm for Feature Selection.
    Chromosome: Binary vector C in {0, 1}^M representing active clinical features.
    Operators:
        - Selection: Roulette Wheel & Tournament Selection
        - Crossover: Single-Point & Uniform Crossover
        - Mutation: Bit-flip mutation
        - Elitism: Preserves top K chromosomes
    """

    def __init__(
        self,
        population_size: int = 24,
        generations: int = 25,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.05,
        parsimony_penalty: float = 0.08,
        selection_method: str = "tournament",  # 'tournament' or 'roulette'
        tournament_size: int = 3,
        elitism_count: int = 2,
        random_state: int = 42,
    ):
        self.pop_size = population_size
        self.generations = generations
        self.pc = crossover_rate
        self.pm = mutation_rate
        self.parsimony_penalty = parsimony_penalty
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.feature_names: List[str] = []
        self.history: List[Dict[str, Any]] = []
        self.best_chromosome: Optional[np.ndarray] = None
        self.best_fitness: float = -1e9
        self.best_accuracy: float = 0.0

    def _init_population(self, num_features: int) -> np.ndarray:
        """Initializes random binary population ensuring at least one feature active per individual."""
        pop = self.rng.choice([0, 1], size=(self.pop_size, num_features), p=[0.5, 0.5])
        # Ensure non-empty
        for i in range(self.pop_size):
            if np.sum(pop[i]) == 0:
                pop[i, self.rng.randint(0, num_features)] = 1
        return pop

    def fitness_function(self, chromosome: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Multi-objective fitness:
            Fitness(C) = CV_Accuracy(C) - gamma * (SelectedFeatures / TotalFeatures)
        """
        selected_indices = np.where(chromosome == 1)[0]
        if len(selected_indices) == 0:
            return 0.0, 0.0

        X_sub = X[:, selected_indices]
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=200, random_state=self.random_state)
        )

        try:
            scores = cross_val_score(clf, X_sub, y, cv=3, scoring="accuracy")
            cv_acc = float(np.mean(scores))
        except Exception:
            cv_acc = 0.50

        num_feat = len(chromosome)
        feature_ratio = len(selected_indices) / num_feat
        fitness = cv_acc - (self.parsimony_penalty * feature_ratio)
        return float(fitness), cv_acc

    def roulette_wheel_selection(self, population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        """Fitness-proportional selection."""
        # Shift positive
        shifted_fit = fitnesses - np.min(fitnesses) + 1e-4
        probs = shifted_fit / np.sum(shifted_fit)
        idx = self.rng.choice(len(population), p=probs)
        return population[idx].copy()

    def tournament_selection(self, population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        """Tournament selection with specified tournament size k."""
        competitors_idx = self.rng.choice(len(population), size=self.tournament_size, replace=False)
        best_comp_idx = competitors_idx[np.argmax(fitnesses[competitors_idx])]
        return population[best_comp_idx].copy()

    def select_parent(self, population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        if self.selection_method == "roulette":
            return self.roulette_wheel_selection(population, fitnesses)
        return self.tournament_selection(population, fitnesses)

    def single_point_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single-point crossover operator."""
        if self.rng.rand() > self.pc or len(parent1) <= 1:
            return parent1.copy(), parent2.copy()

        point = self.rng.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def bit_flip_mutation(self, chromosome: np.ndarray) -> np.ndarray:
        """Bit-flip mutation operator with probability pm per gene."""
        mutated = chromosome.copy()
        for gene_idx in range(len(mutated)):
            if self.rng.rand() < self.pm:
                mutated[gene_idx] = 1 - mutated[gene_idx]

        # Prevent degenerate all-zero chromosome
        if np.sum(mutated) == 0:
            mutated[self.rng.randint(0, len(mutated))] = 1
        return mutated

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Runs the Genetic Algorithm across all generations."""
        self.feature_names = feature_names
        num_features = X.shape[1]
        population = self._init_population(num_features)

        self.history = []
        self.best_fitness = -1e9
        self.best_chromosome = None
        self.best_accuracy = 0.0

        for gen in range(1, self.generations + 1):
            evals = [self.fitness_function(ind, X, y) for ind in population]
            fitnesses = np.array([e[0] for e in evals])
            accuracies = np.array([e[1] for e in evals])

            gen_best_idx = int(np.argmax(fitnesses))
            gen_best_fit = float(fitnesses[gen_best_idx])
            gen_best_acc = float(accuracies[gen_best_idx])
            gen_avg_fit = float(np.mean(fitnesses))

            if gen_best_fit > self.best_fitness:
                self.best_fitness = gen_best_fit
                self.best_accuracy = gen_best_acc
                self.best_chromosome = population[gen_best_idx].copy()

            self.history.append({
                "generation": gen,
                "best_fitness": gen_best_fit,
                "avg_fitness": gen_avg_fit,
                "best_accuracy": gen_best_acc,
                "num_features_selected": int(np.sum(population[gen_best_idx])),
            })

            # Next Generation Formation with Elitism
            sorted_indices = np.argsort(fitnesses)[::-1]
            new_population = []

            # Carry over elites unchanged
            for e_idx in range(self.elitism_count):
                new_population.append(population[sorted_indices[e_idx]].copy())

            # Breed remaining offspring
            while len(new_population) < self.pop_size:
                p1 = self.select_parent(population, fitnesses)
                p2 = self.select_parent(population, fitnesses)
                c1, c2 = self.single_point_crossover(p1, p2)
                c1 = self.bit_flip_mutation(c1)
                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    c2 = self.bit_flip_mutation(c2)
                    new_population.append(c2)

            population = np.array(new_population)

        selected_indices = np.where(self.best_chromosome == 1)[0]
        selected_features = [self.feature_names[i] for i in selected_indices]

        return {
            "best_chromosome": self.best_chromosome.tolist(),
            "best_fitness": self.best_fitness,
            "best_cv_accuracy": self.best_accuracy,
            "selected_features": selected_features,
            "selected_count": len(selected_features),
            "history": self.history,
        }

    @staticmethod
    def schema_theorem_summary() -> str:
        """Holland's Schema Theorem summary for viva defense."""
        return (
            "Holland's Schema Theorem (Fundamental Theorem of Genetic Algorithms): "
            "Short, low-order schemata with above-average fitness increase exponentially in frequency "
            "in subsequent generations: m(H, t+1) >= m(H, t) * (f(H) / f_avg) * [1 - p_c * (delta(H) / (L-1)) - o(H) * p_m]."
        )
