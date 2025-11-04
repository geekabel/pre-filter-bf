"""Multi-sensor fusion using greedy pairwise-GNN + clustering for T2T.

This module wraps the greedy assignment routine from `greedy_multidim.py`
so that we:
1. Form clusters with at most one track per sensor (many-to-one association)
2. Fuse the contributing tracks within each cluster using Covariance Intersection
3. Preserve full provenance so downstream evaluation can reason about couples
"""

from typing import Dict, List, Tuple

import numpy as np

from greedy_multidim import greedy_t2ta
from config import SENSOR_TYPES
from ds_lima import ExistenceBBA, combine_multi_sensor_masses

T2T_NEES_GATE = 5.99  # χ²(2 dof, 95%) for position-only NEES


class TrackCluster:
    """Cluster of tracks from multiple sensors representing the same object."""

    def __init__(self, cluster_id: int):
        self.cluster_id = cluster_id
        # Map sensor key -> full track info dict
        # old Dict[str, dict]
        self.tracks: Dict[str, List[dict]] = {}
        self.fused_state = None
        self.fused_cov = None
        self.fused_existence = None
        self.diagnostics: Dict[str, object] = {}

    def add_track(self, sensor_key: str, track_info: dict):
        """Add a track to this cluster."""
        # if sensor_key in self.tracks:
        #     return
        self.tracks.setdefault(sensor_key,[]).append(track_info)

    def num_tracks(self) -> int:
        """Number of contributing tracks."""
        return sum(len(lst) for lst in self.tracks.values())

    def get_sensor_ids(self) -> List[str]:
        """Return ordered list of sensor keys in this cluster."""
        return list(self.tracks.keys())

    def compute_diagnostics(self):
        """Analyse raw sensor tracks before fusion to flag ghosts/mis-associations."""
        track_summaries = []
        beliefs = []
        assoc_ids = []
        sensor_types = []

        persistent_tokens = []

        for sensor_key, track_list in self.tracks.items():
            for track in track_list:
                existence = track.get('existence', {}) or {}
                belief = existence.get('belief')
                if belief is None and 'm_E' in existence:
                    belief = existence['m_E']
                if belief is not None:
                    beliefs.append(float(belief))

                assoc_id = track.get('associated_object_id')
                if assoc_id is not None:
                    assoc_ids.append(assoc_id)

                sensor_type = track.get('sensor_type', sensor_key)
                sensor_types.append(sensor_type)

                track_summaries.append({
                    'sensor_key': sensor_key,
                    'sensor_type': sensor_type,
                    'track_id': track.get('track_id'),
                    'status': track.get('status'),
                    'associated_object_id': assoc_id,
                    'belief': belief,
                })
                persistent_tokens.append(f"{sensor_key}:{track.get('track_id')}")

        unique_ids = sorted(set(assoc_ids))
        persistent_id = "|".join(sorted(persistent_tokens)) if persistent_tokens else f"cluster_{self.cluster_id}"
        diagnostics = {
            'cluster_id': self.cluster_id,
            'persistent_id': persistent_id,
            'num_tracks': self.num_tracks(),
            'sensor_keys': list(self.tracks.keys()),
            'sensor_types': sensor_types,
            'track_summaries': track_summaries,
            'associated_object_ids': assoc_ids,
            'unique_object_ids': unique_ids,
            'consensus_object_id': unique_ids[0] if len(unique_ids) == 1 else None,
            'is_ghost': len(assoc_ids) == 0,
            'is_mis_association': len(unique_ids) > 1,
            'avg_belief': float(np.mean(beliefs)) if beliefs else 0.0,
            'max_belief': float(np.max(beliefs)) if beliefs else 0.0,
        }
        self.diagnostics = diagnostics

    def fuse_tracks(self):
        """Fuse tracks: fusion-level DS existence first, then CI state fusion."""
        if self.num_tracks() == 0:
            return
        all_tracks = [t for lst in self.tracks.values() for t in lst]

        existence_bbas = [
            ExistenceBBA(t['existence']['m_E'], t['existence']['m_NE'], t['existence']['m_U'])
            for t in all_tracks
        ]
        self.fused_existence = combine_multi_sensor_masses(existence_bbas)

        if len(all_tracks) == 1:
            track = all_tracks[0]
            self.fused_state = track['state'].copy()
            self.fused_cov = track['covariance'].copy()
            return

        states = [t['state'] for t in all_tracks]
        covs = [t['covariance'] for t in all_tracks]
        self.fused_state, self.fused_cov = _fuse_with_covariance_intersection(states, covs)


def _fuse_with_covariance_intersection(states: List[np.ndarray],
                                        covs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse multiple track estimates using equal-weight Covariance Intersection."""
    if len(states) == 0:
        raise ValueError("Cannot fuse empty list")
    if len(states) == 1:
        return states[0].copy(), covs[0].copy()

    omega = 1.0 / len(states)
    dim = states[0].shape[0]

    P_fused_inv = np.zeros((dim, dim))
    weighted_info = np.zeros(dim)

    for state, cov in zip(states, covs):
        try:
            P_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Skip singular covariances; they convey no extra information
            continue
        P_fused_inv += omega * P_inv
        weighted_info += omega * (P_inv @ state)

    try:
        P_fused = np.linalg.inv(P_fused_inv)
        x_fused = P_fused @ weighted_info
    except np.linalg.LinAlgError:
        # Fall back to simple averages if precision matrix is singular
        x_fused = np.mean(states, axis=0)
        P_fused = np.mean(covs, axis=0)

    return x_fused, P_fused


def _sensor_key_mapping(sensor_tracks: Dict[str, List[dict]]) -> Dict[str, int]:
    """Create a contiguous mapping sensor_key -> integer index for clustering."""
    sensor_keys = list(sensor_tracks.keys())
    return {key: idx for idx, key in enumerate(sensor_keys)}


def _extract_belief(track: dict) -> float:
    existence = track.get('existence', {}) or {}
    if 'belief' in existence and existence['belief'] is not None:
        return float(existence['belief'])
    if 'm_E' in existence and existence['m_E'] is not None:
        return float(existence['m_E'])
    return 0.0


def _prepare_track_matrix(sensor_tracks: Dict[str, List[dict]]):
    """Build the feature matrix expected by greedy_t2ta plus provenance info."""
    # sort track by belief
    sensor_to_idx = _sensor_key_mapping(sensor_tracks)
    feature_rows: List[List[float]] = []
    track_records: List[dict] = []

    for sensor_key, tracks in sensor_tracks.items():
        sensor_idx = sensor_to_idx[sensor_key]
        sorted_tracks = sorted(
            tracks,
            key=lambda tr: (
                -_extract_belief(tr),
                tr.get('track_id', -1),
                tr.get('age', 0)
            )
        )
        for track in sorted_tracks:
            state = track.get('state')
            covariance = track.get('covariance')
            if state is None or covariance is None:
                continue
            state_vec = np.asarray(state)
            if state_vec.size < 2:
                continue
            feature_rows.append([float(state_vec[0]), float(state_vec[1]), float(sensor_idx)])
            track_records.append({
                'sensor_key': sensor_key,
                'sensor_index': sensor_idx,
                'track': track
            })

    if not feature_rows:
        return np.empty((0, 3), dtype=float), track_records

    return np.asarray(feature_rows, dtype=float), track_records


def _mahalanobis_squared_t2t(track_i: dict, track_j: dict) -> float:
    """Squared Mahalanobis distance (NEES) using 2D position components."""
    state_i = np.asarray(track_i['state'])
    state_j = np.asarray(track_j['state'])
    cov_i = np.asarray(track_i['covariance'])
    cov_j = np.asarray(track_j['covariance'])

    if state_i.shape[0] < 2 or state_j.shape[0] < 2:
        return float('inf')

    pos_i = state_i[:2]
    pos_j = state_j[:2]
    cov_pos_i = cov_i[:2, :2]
    cov_pos_j = cov_j[:2, :2]

    S = cov_pos_i + cov_pos_j
    diff = pos_i - pos_j

    try:
        S_inv = np.linalg.inv(S)
        return float(diff.T @ S_inv @ diff)
    except np.linalg.LinAlgError:
        return float('inf')


def _build_mahalanobis_distance_matrix(track_records: List[dict]) -> np.ndarray:
    """Construct a symmetric matrix of squared Mahalanobis distances."""
    num_tracks = len(track_records)
    dist_matrix = np.full((num_tracks, num_tracks), np.inf, dtype=float)

    for i in range(num_tracks):
        track_i = track_records[i]['track']
        for j in range(i + 1, num_tracks):
            track_j = track_records[j]['track']

            # Same-sensor blocking removed (redundance)
            # Object ID mismatch logic that prevents false associations
            # sensor_i = track_records[i]['sensor_key']
            # sensor_j = track_records[j]['sensor_key']
            # obj_i = track_i.get('associated_object_id')
            # obj_j = track_j.get('associated_object_id')
            # if obj_i is not None and obj_j is not None and obj_i != obj_j:
            #     continue

            dist = _mahalanobis_squared_t2t(track_i, track_j)
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    return dist_matrix


def _cluster_with_greedy_t2ta(sensor_tracks: Dict[str, List[dict]]) -> List[TrackCluster]:
    """Run greedy_t2ta and convert assignments into TrackCluster objects."""
    feature_matrix, track_records = _prepare_track_matrix(sensor_tracks)

    if len(track_records) == 0:
        return []

    distance_matrix = _build_mahalanobis_distance_matrix(track_records)

    assignments = greedy_t2ta(
        feature_matrix,
        distance_matrix=distance_matrix,
        max_distance=T2T_NEES_GATE,
        allow_same_sensor=True,
        cov=None
    ).astype(int)

    cluster_lookup: Dict[int, int] = {}
    clusters: Dict[int, TrackCluster] = {}
    next_cluster_id = 0

    for idx, label in enumerate(assignments):
        if label > 0:
            if label not in cluster_lookup:
                cluster_lookup[label] = next_cluster_id
                next_cluster_id += 1
            cluster_id = cluster_lookup[label]
        else:
            cluster_id = next_cluster_id
            next_cluster_id += 1

        if cluster_id not in clusters:
            clusters[cluster_id] = TrackCluster(cluster_id)

        record = track_records[idx]
        clusters[cluster_id].add_track(record['sensor_key'], record['track'])
    # ordered clusters
    ordered_clusters = [clusters[cid] for cid in sorted(clusters.keys())]
    for cluster in ordered_clusters:
        cluster.compute_diagnostics()
        cluster.fuse_tracks()

    return ordered_clusters


def get_fused_tracks(clusters: List[TrackCluster]) -> List[dict]:
    """Extract fused track estimates with provenance for downstream evaluation."""
    fused_tracks = []

    for cluster in clusters:
        if cluster.fused_state is None or cluster.fused_existence is None:
            continue

        source_tracks = []
        for sensor_key, track_list in cluster.tracks.items():
            for track_info in track_list:
                source_tracks.append({
                    'cluster_sensor_key': sensor_key,
                    'track_id': track_info.get('track_id'),
                    'sensor_id': track_info.get('sensor_id'),
                    'sensor_type': track_info.get('sensor_type', sensor_key),
                    'state': track_info.get('state'),
                    'position': track_info.get('position'),
                    'covariance': track_info.get('covariance'),
                    'existence': track_info.get('existence'),
                    'status': track_info.get('status'),
                    'associated_object_id': track_info.get('associated_object_id')
                })

        sensor_ids = cluster.get_sensor_ids()
        sensor_types = [
            cluster.tracks[s][0].get('sensor_type', s) for s in sensor_ids
        ]

        fused_tracks.append({
            'cluster_id': cluster.cluster_id,
            'persistent_id': cluster.diagnostics.get('persistent_id'),
            'state': cluster.fused_state,
            'position': cluster.fused_state[:2],
            'covariance': cluster.fused_cov,
            'existence': {
                'm_E': cluster.fused_existence.m_E,
                'm_NE': cluster.fused_existence.m_NE,
                'm_U': cluster.fused_existence.m_U,
                'belief': cluster.fused_existence.belief(),
                'pignistic': cluster.fused_existence.pignistic()
            },
            'num_sources': cluster.num_tracks(),
            'num_sensors': cluster.num_tracks(),
            'sensor_ids': sensor_ids,
            'sensor_types': sensor_types,
            'source_tracks': source_tracks,
            'diagnostics': cluster.diagnostics
        })

    return fused_tracks


class MultiSensorFusion:
    """Multi-sensor fusion system for a vehicle leveraging greedy_t2ta clustering."""

    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.fused_tracks: List[dict] = []
        self.clusters: List[TrackCluster] = []

    def fuse_sensor_tracks(self, sensor_tracks_dict):
        """Fuse tracks from all sensors using the greedy multidimensional assignment."""
        if isinstance(sensor_tracks_dict, list):
            mapped_tracks = {}
            for i, tracks in enumerate(sensor_tracks_dict):
                sensor_key = SENSOR_TYPES[i] if i < len(SENSOR_TYPES) else f"sensor_{i}"
                mapped_tracks[sensor_key] = tracks
            sensor_tracks_dict = mapped_tracks

        self.clusters = _cluster_with_greedy_t2ta(sensor_tracks_dict)
        self.fused_tracks = get_fused_tracks(self.clusters)
        return self.fused_tracks

    def get_fused_tracks(self):
        return self.fused_tracks

    def get_clusters(self):
        return self.clusters

    def get_num_fused_tracks(self):
        return len(self.fused_tracks)

    def get_cluster_diagnostics(self):
        return [cluster.diagnostics for cluster in self.clusters]
