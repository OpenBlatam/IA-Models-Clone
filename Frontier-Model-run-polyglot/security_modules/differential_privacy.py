# differential_privacy.py
# Basado en: Abadi et al. 2016 (Deep Learning with DP), Mironov 2017 (Rényi DP)

import math
import random
from typing import Optional, Tuple

class DifferentialPrivacy:
    """
    Privacidad diferencial para datos y gradientes.
    Implementa DP-SGD con Rényi DP accountant.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, max_grad_norm: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self._noise_std = self._compute_noise_scale()

    def _compute_noise_scale(self) -> float:
        """
        Calcula la escala de ruido Gaussian para alcanzar (epsilon, delta)-DP.
        Usa aproximación analítica para Rényi DP.
        """
        if self.epsilon < 0 or self.delta <= 0 or self.max_grad_norm <= 0:
            raise ValueError("Epsilon, delta y max_grad_norm deben ser > 0")
        # Fórmula simplificada: sigma = sqrt(2 * ln(1.25/delta)) / epsilon
        sigma = (self.max_grad_norm * math.sqrt(2 * math.log(1.25 / self.delta))) / self.epsilon
        return sigma

    def clip_gradient(self, gradient: list, max_norm: Optional[float] = None) -> list:
        """
        Recorta gradiente L2 a max_norm.
        """
        norm = max_norm or self.max_grad_norm
        if norm <= 0:
            return gradient
        total_norm = math.sqrt(sum(g**2 for g in gradient))
        if total_norm > norm:
            scale = norm / total_norm
            return [g * scale for g in gradient]
        return gradient

    def add_noise(self, gradient: list, sensitivity: Optional[float] = None) -> list:
        """
        Añade ruido Gaussiano al gradiente.
        """
        sens = sensitivity or self.max_grad_norm
        noise_std = self._noise_std * sens
        return [g + random.gauss(0, noise_std) for g in gradient]

    def apply_dp(self, gradient: list) -> list:
        """
        Aplica clipping + ruido para DP-SGD.
        """
        clipped = self.clip_gradient(gradient)
        noisy = self.add_noise(clipped)
        return noisy

    def compute_renyi_dp(self, steps: int, batch_size: int, dataset_size: int) -> Tuple[float, float]:
        """
        Calcula Rényi DP después de múltiples pasos.
        Retorna: (epsilon_rdp, delta_actual)
        """
        sampling_rate = batch_size / dataset_size
        # Aproximación RDP: epsilon_k = k * (alpha-1) * q^2 * sigma^(-2) / 2
        alpha = 2.0  # Orden de Rényi
        q = sampling_rate
        sigma = self._noise_std / self.max_grad_norm if self.max_grad_norm > 0 else 1.0
        epsilon_k = steps * (alpha - 1) * (q ** 2) / (2 * sigma ** 2)
        delta_actual = min(1.0, steps * q * math.exp(-(alpha - 1) * (self.epsilon - epsilon_k) / alpha))
        return epsilon_k, delta_actual

    def get_parameters(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "max_grad_norm": self.max_grad_norm,
            "noise_std": self._noise_std
        }
