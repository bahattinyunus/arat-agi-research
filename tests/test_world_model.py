import pytest
import numpy as np
from core_cognition.world_model import (
    RecurrentStateSpaceModel,
    EpistemicCuriosityEngine
)

def test_rssm_step_and_kl_divergence():
    obs_dim = 6
    act_dim = 2
    rssm = RecurrentStateSpaceModel(obs_dim=obs_dim, action_dim=act_dim, deter_dim=16, stoch_dim=8)

    obs = np.array([1.0, 0.5, -0.2, 0.8, 0.0, -1.0])
    act = np.array([0.5, 0.1])

    res = rssm.step(obs, act)

    assert "h" in res
    assert "z" in res
    assert res["h"].shape == (16,)
    assert res["z"].shape == (8,)
    assert res["kl_divergence"] >= 0.0
    assert res["recon_error"] >= 0.0
    assert res["reconstruction"].shape == (obs_dim,)

def test_rssm_imagination_rollout():
    rssm = RecurrentStateSpaceModel(obs_dim=4, action_dim=2, deter_dim=16, stoch_dim=8)
    action_seq = np.zeros((10, 2))
    action_seq[:, 0] = 1.0 # Drive forward

    imagined_obs = rssm.rollout_imagination(action_seq)
    assert imagined_obs.shape == (10, 4)

def test_epistemic_curiosity_engine():
    engine = EpistemicCuriosityEngine(beta_kl=1.0, beta_recon=0.5)

    r_intrinsic = engine.compute_intrinsic_reward(kl_divergence=0.8, recon_error=0.4)
    # Expected: 1.0 * 0.8 + 0.5 * 0.4 = 1.0
    assert np.isclose(r_intrinsic, 1.0, atol=1e-5)

    # Blended reward: 70% extrinsic (10.0), 30% intrinsic (1.0) -> 7.0 + 0.3 = 7.3
    r_blended = engine.blend_reward(extrinsic_reward=10.0, intrinsic_reward=r_intrinsic, curiosity_weight=0.3)
    assert np.isclose(r_blended, 7.3, atol=1e-5)
