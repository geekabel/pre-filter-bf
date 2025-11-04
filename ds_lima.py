"""Lima's sensor-level DS existence reasoning implementation.

This implements the four operations from Lima's thesis:
1. Birth - Initialize tracks from unmatched detections
2. Temporal discounting (half-life) - Decay existence evidence over time
3. Update on association - Boost existence with constant mass
4. Deletion - Remove tracks when unknown mass exceeds threshold
"""

import numpy as np
from config import E_BIRTH, E_FORGET, T_EXISTS_HALF, E_UPDATE, DT, TAU_CONF, TAU_HOLD


class ExistenceBBA:
    """Basic Belief Assignment for existence hypothesis.

    Maintains masses for:
    - m_E: mass for existence (E)
    - m_NE: mass for non-existence (¬E)
    - m_U: mass for unknown (Ω)

    Always: m_E + m_NE + m_U = 1
    """

    def __init__(self, m_E=0.0, m_NE=0.0, m_U=1.0):
        """Initialize BBA.

        Args:
            m_E: Mass for existence
            m_NE: Mass for non-existence
            m_U: Mass for unknown
        """
        self.m_E = m_E
        self.m_NE = m_NE
        self.m_U = m_U
        self._normalize()

    def _normalize(self):
        """Ensure masses sum to 1."""
        total = self.m_E + self.m_NE + self.m_U
        if total > 0:
            self.m_E /= total
            self.m_NE /= total
            self.m_U /= total
        else:
            # Total ignorance
            self.m_E = 0.0
            self.m_NE = 0.0
            self.m_U = 1.0

    def belief(self):
        """Calculate Belief in existence: Bel(E) = m_E."""
        return self.m_E

    def plausibility(self):
        """Calculate Plausibility in existence: Pl(E) = m_E + m_U."""
        return self.m_E + self.m_U

    def pignistic(self):
        """Calculate Pignistic probability: BetP(E) = m_E + 0.5*m_U."""
        return self.m_E + 0.5 * self.m_U

    def copy(self):
        """Create a copy of this BBA."""
        return ExistenceBBA(self.m_E, self.m_NE, self.m_U)


def birth_mass():
    """Create birth mass for new track (Eq. 3.41).

    Returns:
        ExistenceBBA with birth initialization
    """
    return ExistenceBBA(m_E=E_BIRTH, m_NE=0.0, m_U=1.0 - E_BIRTH)


def temporal_discounting(bba, dt=DT):
    """Apply temporal discounting with half-life (Eq. 3.43).

    Args:
        bba: ExistenceBBA to discount
        dt: Time step (default: DT from config)

    Returns:
        New ExistenceBBA after discounting
    """
    # Discount factor: λ = exp(-ln(2) / t_half * dt)
    lambda_val = np.exp(-np.log(2) / T_EXISTS_HALF * dt)

    # Apply discounting (Eq. 3.43)
    m_E_new = lambda_val * bba.m_E
    m_NE_new = lambda_val * bba.m_NE
    m_U_new = 1.0 - m_E_new - m_NE_new

    return ExistenceBBA(m_E_new, m_NE_new, m_U_new)


def update_on_association(bba):
    """Update existence when track associates with detection (Eq. 3.44).

    Uses Dempster's rule to combine with constant boost mass.

    Args:
        bba: Current ExistenceBBA

    Returns:
        New ExistenceBBA after combination
    """
    # Boost mass: m_boost = {E: E_UPDATE, Ω: 1-E_UPDATE}
    m_boost_E = E_UPDATE
    m_boost_U = 1.0 - E_UPDATE

    # Dempster's rule of combination
    # Conflict: K = m1(E)*m2(¬E) + m1(¬E)*m2(E)
    K = bba.m_E * 0.0 + bba.m_NE * m_boost_E

    if K >= 1.0:
        # Total conflict - return current BBA
        return bba.copy()

    # Combined masses (conjunctive combination)
    # m(E) = [m1(E)*m2(E) + m1(E)*m2(Ω) + m1(Ω)*m2(E)] / (1-K)
    # m(¬E) = [m1(¬E)*m2(¬E) + m1(¬E)*m2(Ω) + m1(Ω)*m2(¬E)] / (1-K)
    # m(Ω) = [m1(Ω)*m2(Ω)] / (1-K)

    normalizer = 1.0 / (1.0 - K)

    m_E_new = (bba.m_E * m_boost_E + bba.m_E * m_boost_U + bba.m_U * m_boost_E) * normalizer
    m_NE_new = (bba.m_NE * 0.0 + bba.m_NE * m_boost_U + bba.m_U * 0.0) * normalizer
    m_U_new = (bba.m_U * m_boost_U) * normalizer

    return ExistenceBBA(m_E_new, m_NE_new, m_U_new)


def should_delete(bba):
    """Check if track should be deleted (Eq. 3.42).

    Delete when unknown mass exceeds threshold (too uninformative).

    Args:
        bba: ExistenceBBA to check

    Returns:
        True if track should be deleted
    """
    return bba.m_U > E_FORGET


def get_track_status(bba):
    """Get track status based on belief thresholds.

    Args:
        bba: ExistenceBBA

    Returns:
        'confirmed', 'tentative', or 'decaying'
    """
    belief = bba.belief()

    if belief >= TAU_CONF:
        return 'confirmed'
    elif belief >= TAU_HOLD:
        return 'tentative'
    else:
        return 'decaying'


def combine_multi_sensor_masses(bba_list):
    """Combine existence masses from multiple sensors using Dempster's rule.

    Args:
        bba_list: List of ExistenceBBA objects

    Returns:
        Combined ExistenceBBA
    """
    if len(bba_list) == 0:
        return ExistenceBBA(0.0, 0.0, 1.0)  # Total ignorance

    if len(bba_list) == 1:
        return bba_list[0].copy()

    # Start with first BBA
    result = bba_list[0].copy()

    # Combine with each subsequent BBA
    for bba in bba_list[1:]:
        # Dempster's rule
        K = result.m_E * bba.m_NE + result.m_NE * bba.m_E

        if K >= 1.0:
            # Total conflict - skip this combination
            continue

        normalizer = 1.0 / (1.0 - K)

        m_E_new = (result.m_E * bba.m_E + result.m_E * bba.m_U + result.m_U * bba.m_E) * normalizer
        m_NE_new = (result.m_NE * bba.m_NE + result.m_NE * bba.m_U + result.m_U * bba.m_NE) * normalizer
        m_U_new = (result.m_U * bba.m_U) * normalizer

        result = ExistenceBBA(m_E_new, m_NE_new, m_U_new)

    return result

    # Look at it later
    def cautious_combine(bba):
        m_list = list(bba)

        return ExistenceBBA()
