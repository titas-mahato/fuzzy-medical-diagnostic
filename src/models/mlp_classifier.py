"""
3-Layer Feedforward Multi-Layer Perceptron (MLP) Classifier from scratch.
Trained via Supervised Backpropagation and the Generalized Delta Learning Rule.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


class MLPClassifier:
    """
    3-Layer Feedforward Artificial Neural Network for Cardiovascular Risk Classification.
    Architecture:
        Input Layer (D features) -> Hidden Layer 1 (H1=16, ReLU)
        -> Hidden Layer 2 (H2=8, ReLU) -> Output Layer (K=1, Sigmoid)
    Trained with Supervised Backpropagation, Momentum, and L2 Regularization.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: Tuple[int, int] = (16, 8),
        learning_rate: float = 0.05,
        momentum: float = 0.9,
        l2_reg: float = 0.001,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.h1_dim, self.h2_dim = hidden_dims
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_reg = l2_reg
        self.random_state = random_state

        # Feature normalization state
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

        # Weights and biases (He normal initialization for ReLU)
        rng = np.random.RandomState(random_state)
        self.W1 = rng.randn(self.input_dim, self.h1_dim) * np.sqrt(2.0 / self.input_dim)
        self.b1 = np.zeros((1, self.h1_dim))

        self.W2 = rng.randn(self.h1_dim, self.h2_dim) * np.sqrt(2.0 / self.h1_dim)
        self.b2 = np.zeros((1, self.h2_dim))

        self.W3 = rng.randn(self.h2_dim, 1) * np.sqrt(1.0 / self.h2_dim)
        self.b3 = np.zeros((1, 1))

        # Velocity matrices for momentum
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        self.vW3 = np.zeros_like(self.W3)
        self.vb3 = np.zeros_like(self.b3)

        self.training_history: List[Dict[str, float]] = []

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        return (z > 0.0).astype(float)

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        z_clamped = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z_clamped))

    def fit_scaler(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std < 1e-8] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return X
        return (X - self.mean) / self.std

    def forward(self, X_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward propagation through all 3 layers.
        Returns: (z1, a1, z2, a2, y_hat)
        """
        z1 = np.dot(X_norm, self.W1) + self.b1
        a1 = self.relu(z1)

        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.relu(z2)

        z3 = np.dot(a2, self.W3) + self.b3
        y_hat = self.sigmoid(z3)

        return z1, a1, z2, a2, y_hat

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_norm = self.transform(X)
        _, _, _, _, y_hat = self.forward(X_norm)
        return y_hat.flatten()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary cross-entropy loss with L2 weight regularization."""
        eps = 1e-9
        y_clamped = np.clip(y_pred, eps, 1.0 - eps)
        bce = -np.mean(y_true * np.log(y_clamped) + (1.0 - y_true) * np.log(1.0 - y_clamped))
        l2 = (self.l2_reg / (2.0 * len(y_true))) * (
            np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2) + np.sum(self.W3 ** 2)
        )
        return float(bce + l2)

    def train_step(self, X_batch: np.ndarray, y_batch: np.ndarray):
        """
        One gradient backpropagation step using the Generalized Delta Learning Rule.
        """
        N = X_batch.shape[0]
        y_col = y_batch.reshape(-1, 1)

        # 1. Forward Pass
        z1, a1, z2, a2, y_hat = self.forward(X_batch)

        # 2. Delta Rule at Output Layer (Sigmoid + BCE gradient)
        # dL/dz3 = y_hat - y
        delta3 = y_hat - y_col

        dW3 = (1.0 / N) * np.dot(a2.T, delta3) + (self.l2_reg / N) * self.W3
        db3 = (1.0 / N) * np.sum(delta3, axis=0, keepdims=True)

        # 3. Delta Backpropagation to Hidden Layer 2
        # dL/dz2 = (delta3 * W3.T) * relu'(z2)
        delta2 = np.dot(delta3, self.W3.T) * self.relu_derivative(z2)

        dW2 = (1.0 / N) * np.dot(a1.T, delta2) + (self.l2_reg / N) * self.W2
        db2 = (1.0 / N) * np.sum(delta2, axis=0, keepdims=True)

        # 4. Delta Backpropagation to Hidden Layer 1
        # dL/dz1 = (delta2 * W2.T) * relu'(z1)
        delta1 = np.dot(delta2, self.W2.T) * self.relu_derivative(z1)

        dW1 = (1.0 / N) * np.dot(X_batch.T, delta1) + (self.l2_reg / N) * self.W1
        db1 = (1.0 / N) * np.sum(delta1, axis=0, keepdims=True)

        # 5. Momentum Weight & Bias Updates
        self.vW3 = self.momentum * self.vW3 + (1.0 - self.momentum) * dW3
        self.vb3 = self.momentum * self.vb3 + (1.0 - self.momentum) * db3
        self.W3 -= self.learning_rate * self.vW3
        self.b3 -= self.learning_rate * self.vb3

        self.vW2 = self.momentum * self.vW2 + (1.0 - self.momentum) * dW2
        self.vb2 = self.momentum * self.vb2 + (1.0 - self.momentum) * db2
        self.W2 -= self.learning_rate * self.vW2
        self.b2 -= self.learning_rate * self.vb2

        self.vW1 = self.momentum * self.vW1 + (1.0 - self.momentum) * dW1
        self.vb1 = self.momentum * self.vb1 + (1.0 - self.momentum) * db1
        self.W1 -= self.learning_rate * self.vW1
        self.b1 -= self.learning_rate * self.vb1

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 250,
        batch_size: int = 32,
    ):
        """Trains the MLP over specified epochs tracking training and validation metrics."""
        self.fit_scaler(X_train)
        X_tr_norm = self.transform(X_train)
        X_val_norm = self.transform(X_val) if X_val is not None else None

        N = X_tr_norm.shape[0]
        self.training_history = []

        for epoch in range(1, epochs + 1):
            # Shuffle batch
            indices = np.random.permutation(N)
            for start_idx in range(0, N, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                self.train_step(X_tr_norm[batch_idx], y_train[batch_idx])

            # Epoch metrics
            _, _, _, _, train_pred = self.forward(X_tr_norm)
            train_loss = self.compute_loss(y_train.reshape(-1, 1), train_pred)
            train_acc = np.mean((train_pred.flatten() >= 0.5) == y_train)

            epoch_log = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": float(train_acc),
            }

            if X_val_norm is not None and y_val is not None:
                _, _, _, _, val_pred = self.forward(X_val_norm)
                val_loss = self.compute_loss(y_val.reshape(-1, 1), val_pred)
                val_acc = np.mean((val_pred.flatten() >= 0.5) == y_val)
                epoch_log["val_loss"] = val_loss
                epoch_log["val_acc"] = float(val_acc)

            self.training_history.append(epoch_log)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calculates comprehensive academic evaluation metrics."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        y_true = np.asarray(y).flatten()

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))  # Sensitivity
        specificity = tn / max(1, (tn + fp))
        f1_score = (2.0 * precision * recall) / max(1e-8, (precision + recall))

        # Approximate ROC-AUC via rank-sum (Wilcoxon-Mann-Whitney)
        pos_scores = y_proba[y_true == 1]
        neg_scores = y_proba[y_true == 0]
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            roc_auc = float(np.mean([np.mean(p > neg_scores) + 0.5 * np.mean(p == neg_scores) for p in pos_scores]))
        else:
            roc_auc = 0.5

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1_score": float(f1_score),
            "roc_auc": float(roc_auc),
            "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        }


def train_mlp_on_dataset(
    csv_path: str = "D:/Zenith/Projects/fuzzy-medical-diagnostic/data/heart.csv",
    feature_cols: Optional[List[str]] = None,
    epochs: int = 220,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[MLPClassifier, Dict[str, Any], pd.DataFrame]:
    """Helper to train the MLP on the UCI Heart Disease dataset."""
    df = pd.read_csv(csv_path)

    if feature_cols is None:
        # Default 5 continuous clinical vitals matching fuzzy inputs
        feature_cols = ["age", "trestbps", "chol", "fbs", "thalach"]

    X = df[feature_cols].values
    y = df["target"].values

    np.random.seed(random_state)
    n = len(df)
    perm = np.random.permutation(n)
    split_idx = int(n * (1.0 - test_size))

    train_idx, test_idx = perm[:split_idx], perm[split_idx:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    mlp = MLPClassifier(input_dim=len(feature_cols), hidden_dims=(16, 8), learning_rate=0.04, random_state=random_state)
    mlp.fit(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=24)
    eval_metrics = mlp.evaluate(X_test, y_test)

    return mlp, eval_metrics, df
