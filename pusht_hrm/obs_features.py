"""
Enhanced observation features for PushT.

Debug analysis showed the model can navigate to the block but can't push
correctly. The raw 5D state (agent_x, agent_y, block_x, block_y, angle)
doesn't explicitly encode the geometric relationships needed for pushing.

This module computes additional features:
- Relative positions (agent-block, block-goal)
- Distance and direction vectors
- Angle differences (block angle vs goal angle)
- Sin/cos of angles for smooth representation
- Pushing geometry hints
"""
import numpy as np
import torch


# Goal pose for PushT (fixed)
GOAL_X, GOAL_Y, GOAL_ANGLE = 256.0, 256.0, np.pi / 4


def compute_geometric_features(obs):
    """
    Compute geometric features from raw observation.

    Input: obs (..., 5) = [agent_x, agent_y, block_x, block_y, block_angle]
    Output: features (..., D) with rich geometric information

    Works with both numpy and torch.
    """
    is_torch = isinstance(obs, torch.Tensor)

    if is_torch:
        sin_fn, cos_fn, atan2_fn = torch.sin, torch.cos, torch.atan2
        sqrt_fn = torch.sqrt
        cat_fn = lambda tensors, dim: torch.cat(tensors, dim=dim)
        stack_fn = lambda tensors, dim: torch.stack(tensors, dim=dim)
        clip_fn = torch.clamp
    else:
        sin_fn, cos_fn, atan2_fn = np.sin, np.cos, np.arctan2
        sqrt_fn = np.sqrt
        cat_fn = lambda arrays, dim: np.concatenate(arrays, axis=dim)
        stack_fn = lambda arrays, dim: np.stack(arrays, axis=dim)
        clip_fn = np.clip

    agent_x = obs[..., 0]
    agent_y = obs[..., 1]
    block_x = obs[..., 2]
    block_y = obs[..., 3]
    block_angle = obs[..., 4]

    # Relative positions
    agent_to_block_x = block_x - agent_x
    agent_to_block_y = block_y - agent_y
    block_to_goal_x = GOAL_X - block_x
    block_to_goal_y = GOAL_Y - block_y
    agent_to_goal_x = GOAL_X - agent_x
    agent_to_goal_y = GOAL_Y - agent_y

    # Distances
    dist_agent_block = sqrt_fn(agent_to_block_x**2 + agent_to_block_y**2 + 1e-6)
    dist_block_goal = sqrt_fn(block_to_goal_x**2 + block_to_goal_y**2 + 1e-6)
    dist_agent_goal = sqrt_fn(agent_to_goal_x**2 + agent_to_goal_y**2 + 1e-6)

    # Normalized direction vectors
    dir_a2b_x = agent_to_block_x / (dist_agent_block + 1e-6)
    dir_a2b_y = agent_to_block_y / (dist_agent_block + 1e-6)
    dir_b2g_x = block_to_goal_x / (dist_block_goal + 1e-6)
    dir_b2g_y = block_to_goal_y / (dist_block_goal + 1e-6)

    # Angle features
    angle_diff = block_angle - GOAL_ANGLE
    sin_angle = sin_fn(block_angle)
    cos_angle = cos_fn(block_angle)
    sin_diff = sin_fn(angle_diff)
    cos_diff = cos_fn(angle_diff)

    # Angle from agent to block (polar)
    angle_a2b = atan2_fn(agent_to_block_y, agent_to_block_x)
    sin_a2b = sin_fn(angle_a2b)
    cos_a2b = cos_fn(angle_a2b)

    # Angle from block to goal (polar)
    angle_b2g = atan2_fn(block_to_goal_y, block_to_goal_x)
    sin_b2g = sin_fn(angle_b2g)
    cos_b2g = cos_fn(angle_b2g)

    # Pushing geometry: angle between agent-block line and block-goal line
    # This tells the agent which side to approach from
    push_angle_diff = angle_a2b - angle_b2g
    sin_push = sin_fn(push_angle_diff)
    cos_push = cos_fn(push_angle_diff)

    # Normalize distances to [0, 1] range (max distance ~500 in 512x512 space)
    norm_dist_ab = dist_agent_block / 500.0
    norm_dist_bg = dist_block_goal / 500.0
    norm_dist_ag = dist_agent_goal / 500.0

    # Normalize positions to [-1, 1]
    norm_agent_x = (agent_x - 256) / 256
    norm_agent_y = (agent_y - 256) / 256
    norm_block_x = (block_x - 256) / 256
    norm_block_y = (block_y - 256) / 256

    features = stack_fn([
        # Normalized raw positions (4)
        norm_agent_x, norm_agent_y, norm_block_x, norm_block_y,
        # Block angle features (4)
        sin_angle, cos_angle, sin_diff, cos_diff,
        # Distances (3)
        norm_dist_ab, norm_dist_bg, norm_dist_ag,
        # Direction: agent to block (2)
        dir_a2b_x, dir_a2b_y,
        # Direction: block to goal (2)
        dir_b2g_x, dir_b2g_y,
        # Polar angles (4)
        sin_a2b, cos_a2b, sin_b2g, cos_b2g,
        # Pushing geometry (2)
        sin_push, cos_push,
    ], dim=-1)

    return features  # (..., 21)


GEOMETRIC_FEATURE_DIM = 21


def augment_obs_with_geometry(obs_raw):
    """
    Take raw obs (agent_x, agent_y, block_x, block_y, angle) and return
    augmented observation with geometric features appended.

    Input: (..., 5) raw obs
    Output: (..., 26) augmented obs (5 raw + 21 geometric)
    """
    geo = compute_geometric_features(obs_raw)
    if isinstance(obs_raw, torch.Tensor):
        return torch.cat([obs_raw, geo], dim=-1)
    return np.concatenate([obs_raw, geo], axis=-1)
