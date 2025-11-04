import numpy as np
from scipy.spatial.distance import cdist


import numpy as np
from scipy.spatial.distance import cdist

def greedy_t2ta(tracks, old_version=False, distance_matrix=None, max_distance=10.0,
                allow_same_sensor=False, cov=None):
    """
    Adapted Greedy T2TA (your original API preserved).
    - allow_same_sensor=False  -> original behavior (forbid two tracks from same sensor in a cluster)
    - allow_same_sensor=True   -> allow multiple tracks from the same sensor
    - cov=None                 -> Euclidean distance (original)
      cov = (2x2) np.array     -> Mahalanobis with VI = inv(cov) on tracks[:, :2]
    """
    num_sensors = len(set(tracks[:, -1]))
    num_tracks = tracks.shape[0]
    max_dist = 9999


    # --- pairwise distance
    if distance_matrix is not None:
        dist = distance_matrix.copy()
    else:
        if cov is not None:
            # Mahalanobis on x,y using provided covariance
            VI = np.linalg.inv(np.asarray(cov, dtype=float))
            dist = cdist(tracks[:, :2], tracks[:, :2], metric='mahalanobis', VI=VI)
        else:
             # NEES maha
            dist = cdist(tracks[:, :2], tracks[:, :2])

    # --- upper triangle + gating
    dist[np.triu_indices(num_tracks)] = max_dist
    dist[dist > max_distance] = max_dist

    # --- forbid same-sensor pairs (only if NOT allowing them)
    if not allow_same_sensor:
        # Use unique IDs (works even if sensor ids are not 0..M-1)
        for s in np.unique(tracks[:, -1]):
            idx = tracks[:, -1] == s
            mask = idx[:, None] @ idx[None, :]
            dist[mask] = max_dist

    rows, cols = np.unravel_index(np.argsort(dist.flatten()), shape=dist.shape) # rows[:k], cols[:k]
    joint_asso = np.zeros(num_tracks)  # all singletons

    next_cluster = 1
    for r, c in zip(rows, cols):
        idx_r = joint_asso == joint_asso[r]  # tracks in same cluster as r
        idx_c = joint_asso == joint_asso[c]  # tracks in same cluster as c
        if dist[r, c] >= max_dist:
            continue

        if joint_asso[r] == 0 and joint_asso[c] == 0:  # both singletons
            joint_asso[r] = joint_asso[c] = next_cluster  # new cluster
            next_cluster += 1

        elif joint_asso[r] == 0:  # r is singleton
            # If allowing same-sensor, skip the check; else keep your original condition
            if allow_same_sensor or (tracks[r, -1] not in tracks[idx_c, -1]):
                joint_asso[r] = joint_asso[c]

        elif joint_asso[c] == 0:  # c is singleton
            if allow_same_sensor or (tracks[c, -1] not in tracks[idx_r, -1]):
                joint_asso[c] = joint_asso[r]

        else:  # both in a (different) cluster
            if not old_version and joint_asso[c] != joint_asso[r]:
                # Allow merge regardless of sensor overlap if allow_same_sensor=True
                if allow_same_sensor or set(tracks[idx_c, -1]).isdisjoint(set(tracks[idx_r, -1])):
                    joint_asso[idx_c] = joint_asso[r]  # merge clusters

        # Per-iteration sensor masking (only if NOT allowing same-sensor)
        if not allow_same_sensor:
            tr = tracks[r, -1]
            tc = tracks[c, -1]
            dist[r, tracks[:, -1] == tc] = max_dist
            dist[tracks[:, -1] == tr, c] = max_dist

    return joint_asso
