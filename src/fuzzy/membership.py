"""
Membership Functions for Fuzzy Inference Systems.
Implements Triangular, Trapezoidal, and Gaussian membership curves.
"""

import numpy as np


class MembershipFunction:
    """Base class for all fuzzy membership functions."""

    def __init__(self, name: str):
        self.name = name

    def evaluate(self, x: np.ndarray | float) -> np.ndarray | float:
        raise NotImplementedError("Subclasses must implement evaluate()")

    def __call__(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.evaluate(x)


class TriangularMF(MembershipFunction):
    """
    Triangular Membership Function: trimf(x; a, b, c)
    Parameters:
        a: Left foot (mu = 0)
        b: Peak (mu = 1)
        c: Right foot (mu = 0)
    Formula:
        mu(x) = max(0, min((x - a) / (b - a), (c - x) / (c - b)))
    """

    def __init__(self, name: str, a: float, b: float, c: float):
        super().__init__(name)
        assert a <= b <= c, f"TriangularMF parameters must satisfy a <= b <= c. Got a={a}, b={b}, c={c}"
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def evaluate(self, x: np.ndarray | float) -> np.ndarray | float:
        x_arr = np.asarray(x, dtype=float)
        mu = np.zeros_like(x_arr)

        # Left slope
        if self.b > self.a:
            left_mask = (x_arr >= self.a) & (x_arr <= self.b)
            mu[left_mask] = (x_arr[left_mask] - self.a) / (self.b - self.a)
        else:
            mu[x_arr == self.b] = 1.0

        # Right slope
        if self.c > self.b:
            right_mask = (x_arr > self.b) & (x_arr <= self.c)
            mu[right_mask] = (self.c - x_arr[right_mask]) / (self.c - self.b)
        else:
            mu[x_arr == self.b] = 1.0

        return mu.item() if np.isscalar(x) else mu


class TrapezoidalMF(MembershipFunction):
    """
    Trapezoidal Membership Function: trapmf(x; a, b, c, d)
    Parameters:
        a: Left foot (mu = 0)
        b: Left shoulder (mu = 1)
        c: Right shoulder (mu = 1)
        d: Right foot (mu = 0)
    Formula:
        mu(x) = max(0, min((x - a) / (b - a), 1, (d - x) / (d - c)))
    """

    def __init__(self, name: str, a: float, b: float, c: float, d: float):
        super().__init__(name)
        assert a <= b <= c <= d, f"TrapezoidalMF parameters must satisfy a <= b <= c <= d. Got a={a}, b={b}, c={c}, d={d}"
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)

    def evaluate(self, x: np.ndarray | float) -> np.ndarray | float:
        x_arr = np.asarray(x, dtype=float)
        mu = np.zeros_like(x_arr)

        # Plateau
        plateau_mask = (x_arr >= self.b) & (x_arr <= self.c)
        mu[plateau_mask] = 1.0

        # Left slope
        if self.b > self.a:
            left_mask = (x_arr >= self.a) & (x_arr < self.b)
            mu[left_mask] = (x_arr[left_mask] - self.a) / (self.b - self.a)

        # Right slope
        if self.d > self.c:
            right_mask = (x_arr > self.c) & (x_arr <= self.d)
            mu[right_mask] = (self.d - x_arr[right_mask]) / (self.d - self.c)

        return mu.item() if np.isscalar(x) else mu


class GaussianMF(MembershipFunction):
    """
    Gaussian Membership Function: gaussmf(x; mean, sigma)
    Formula:
        mu(x) = exp(-0.5 * ((x - mean) / sigma)^2)
    """

    def __init__(self, name: str, mean: float, sigma: float):
        super().__init__(name)
        assert sigma > 0, f"GaussianMF sigma must be strictly positive. Got sigma={sigma}"
        self.mean = float(mean)
        self.sigma = float(sigma)

    def evaluate(self, x: np.ndarray | float) -> np.ndarray | float:
        x_arr = np.asarray(x, dtype=float)
        mu = np.exp(-0.5 * ((x_arr - self.mean) / self.sigma) ** 2)
        return mu.item() if np.isscalar(x) else mu
