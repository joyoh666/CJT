import numpy as np

class CJTMRTPrecoder:
    """Unit-norm conjugate matched-filter precoder for one CJT user."""

    def precode(self, h_pred: np.ndarray) -> np.ndarray:
        h_pred = np.asarray(h_pred, dtype=np.complex64)
        norm = np.linalg.norm(h_pred)

        if norm < 1e-12:
            return np.zeros_like(h_pred)
        else:
            return np.conj(h_pred) / norm
        
