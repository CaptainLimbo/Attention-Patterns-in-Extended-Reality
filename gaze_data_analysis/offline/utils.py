import numpy as np


def angular_distance(vecs_1, vecs_2):

    dot = np.einsum("ij,ij->i", vecs_1, vecs_2) / (
        np.linalg.norm(vecs_1, axis=1) * np.linalg.norm(vecs_2, axis=1)
    )
    dot = np.clip(dot, -1, 1)

    return np.abs(np.rad2deg(np.arccos(dot)))
