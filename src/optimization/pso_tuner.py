"""
Particle Swarm Optimization (PSO) for Fuzzy Membership Boundary Tuning.
Optimizes clinical fuzzy sets to minimize diagnostic RMSE against patient outcomes.
Course: CSE315 - Module 5: Evolutionary Algorithms & Swarm Intelligence
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from ..fuzzy.mamdani_fis import MamdaniFIS
from ..fuzzy.membership import TriangularMF, TrapezoidalMF


class Particle:
    """Individual particle within the PSO swarm."""

    def __init__(self, bounds_min: np.ndarray, bounds_max: np.ndarray, rng: np.random.RandomState):
        self.position = rng.uniform(bounds_min, bounds_max)
        self.velocity = rng.uniform(-(bounds_max - bounds_min) * 0.1, (bounds_max - bounds_min) * 0.1)
        self.pbest_position = self.position.copy()
        self.pbest_fitness = float("inf")  # Minimizing RMSE loss
        self.current_fitness = float("inf")


class ParticleSwarmTuner:
    """
    Particle Swarm Optimization (PSO) Engine.
    Tunes 6 key inflection boundaries of the Mamdani FIS:
        x = [BP_Normal_Peak, BP_PreHyp_Peak, BP_High_Peak, Chol_Borderline_Peak, Chol_Dangerous_Foot, Chol_Dangerous_Peak]
    Equations:
        v_i(t+1) = w(t)*v_i(t) + c1*r1*(pBest_i - x_i(t)) + c2*r2*(gBest - x_i(t))
        x_i(t+1) = x_i(t) + v_i(t+1)
    """

    def __init__(
        self,
        num_particles: int = 15,
        max_iterations: int = 20,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 1.8,  # Cognitive acceleration coefficient
        c2: float = 1.8,  # Social acceleration coefficient
        random_state: int = 42,
    ):
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Parameter Search Space Bounds
        # [BP_Norm, BP_Pre, BP_High, Chol_Bord, Chol_Dang_Foot, Chol_Dang_Peak]
        self.bounds_min = np.array([105.0, 125.0, 145.0, 205.0, 220.0, 255.0])
        self.bounds_max = np.array([125.0, 142.0, 168.0, 235.0, 248.0, 290.0])

        self.gbest_position = (self.bounds_min + self.bounds_max) / 2.0
        self.gbest_fitness = float("inf")
        self.convergence_curve: List[Dict[str, float]] = []

    def _apply_params_to_fis(self, fis: MamdaniFIS, params: np.ndarray):
        """Updates the Mamdani FIS membership functions using candidate particle position."""
        bp_norm, bp_pre, bp_high, chol_bord, chol_dang_a, chol_dang_b = params

        # Enforce parameter order monotonicity
        bp_norm = min(bp_norm, bp_pre - 5.0)
        bp_pre = min(bp_pre, bp_high - 5.0)
        chol_dang_a = min(chol_dang_a, chol_dang_b - 5.0)

        # Update Blood Pressure MFs
        fis.inputs["blood_pressure"].terms["Normal"] = TriangularMF(
            "Normal", a=bp_norm - 25.0, b=bp_norm, c=bp_norm + 15.0
        )
        fis.inputs["blood_pressure"].terms["PreHypertensive"] = TriangularMF(
            "PreHypertensive", a=bp_pre - 15.0, b=bp_pre, c=bp_pre + 12.0
        )
        fis.inputs["blood_pressure"].terms["High"] = TriangularMF(
            "High", a=bp_high - 15.0, b=bp_high, c=bp_high + 20.0
        )

        # Update Cholesterol MFs
        fis.inputs["cholesterol"].terms["Borderline"] = TriangularMF(
            "Borderline", a=chol_bord - 30.0, b=chol_bord, c=chol_bord + 30.0
        )
        fis.inputs["cholesterol"].terms["Dangerous"] = TrapezoidalMF(
            "Dangerous", a=chol_dang_a, b=chol_dang_b, c=400.0, d=400.0
        )

    def evaluate_particle(
        self,
        params: np.ndarray,
        val_samples: List[Dict[str, float]],
        y_targets: np.ndarray,
    ) -> float:
        """Evaluates Root Mean Square Error (RMSE) against clinical gold-standard outcomes."""
        temp_fis = MamdaniFIS()
        self._apply_params_to_fis(temp_fis, params)

        predicted_risks = []
        for sample in val_samples:
            res = temp_fis.compute(sample)
            predicted_risks.append(res["risk_score"])

        pred_arr = np.array(predicted_risks)
        rmse = np.sqrt(np.mean((pred_arr - y_targets) ** 2))
        return float(rmse)

    def fit(
        self,
        df: pd.DataFrame,
        sample_size: int = 50,
    ) -> Dict[str, Any]:
        """Runs the PSO optimization loop over the specified number of iterations."""
        # Prepare evaluation patient cohort
        sub_df = df.sample(n=min(sample_size, len(df)), random_state=self.random_state)
        val_samples = []
        for _, row in sub_df.iterrows():
            val_samples.append({
                "blood_pressure": float(row["trestbps"]),
                "cholesterol": float(row["chol"]),
                "glucose": 135.0 if row["fbs"] == 1 else 95.0,
                "heart_rate": float(row["thalach"]),
                "age": float(row["age"]),
            })
        # Target risk: 100% for diseased, 15% for non-diseased
        y_targets = np.where(sub_df["target"].values == 1, 85.0, 20.0)

        # Calculate initial baseline RMSE
        baseline_fis = MamdaniFIS()
        baseline_preds = [baseline_fis.compute(s)["risk_score"] for s in val_samples]
        baseline_rmse = float(np.sqrt(np.mean((np.array(baseline_preds) - y_targets) ** 2)))

        # Initialize Swarm
        particles = [
            Particle(self.bounds_min, self.bounds_max, self.rng)
            for _ in range(self.num_particles)
        ]

        self.gbest_fitness = float("inf")
        self.gbest_position = particles[0].position.copy()
        self.convergence_curve = []

        v_max = (self.bounds_max - self.bounds_min) * 0.20

        for iteration in range(1, self.max_iterations + 1):
            # Time-varying inertia weight decay
            w = self.w_max - iteration * ((self.w_max - self.w_min) / self.max_iterations)

            for p in particles:
                fitness = self.evaluate_particle(p.position, val_samples, y_targets)
                p.current_fitness = fitness

                # Update Personal Best
                if fitness < p.pbest_fitness:
                    p.pbest_fitness = fitness
                    p.pbest_position = p.position.copy()

                # Update Global Best
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = p.position.copy()

            self.convergence_curve.append({
                "iteration": iteration,
                "gbest_rmse": float(self.gbest_fitness),
                "inertia_weight": float(w),
            })

            # Update Velocity & Position
            for p in particles:
                r1 = self.rng.rand(len(p.position))
                r2 = self.rng.rand(len(p.position))

                cognitive = self.c1 * r1 * (p.pbest_position - p.position)
                social = self.c2 * r2 * (self.gbest_position - p.position)

                new_velocity = w * p.velocity + cognitive + social
                new_velocity = np.clip(new_velocity, -v_max, v_max)
                p.velocity = new_velocity

                new_position = p.position + new_velocity
                new_position = np.clip(new_position, self.bounds_min, self.bounds_max)
                p.position = new_position

        tuned_params = {
            "BP_Normal_Peak": float(self.gbest_position[0]),
            "BP_PreHypertensive_Peak": float(self.gbest_position[1]),
            "BP_High_Peak": float(self.gbest_position[2]),
            "Chol_Borderline_Peak": float(self.gbest_position[3]),
            "Chol_Dangerous_Foot": float(self.gbest_position[4]),
            "Chol_Dangerous_Peak": float(self.gbest_position[5]),
        }

        return {
            "baseline_rmse": baseline_rmse,
            "tuned_rmse": self.gbest_fitness,
            "improvement_pct": float(max(0.0, (baseline_rmse - self.gbest_fitness) / baseline_rmse * 100.0)),
            "tuned_parameters": tuned_params,
            "convergence_curve": self.convergence_curve,
        }
