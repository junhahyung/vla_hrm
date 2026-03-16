"""
PushT environment wrapper for evaluation.
Adapted from diffusion_policy's pusht_env.py.
"""
import numpy as np
import torch
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import math


def pymunk_to_shapely(body, shapes):
    geoms = []
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f"Unsupported shape: {shape}")
    return sg.MultiPolygon(geoms)


class PushTEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}

    def __init__(
        self,
        legacy=False,
        block_cog=None,
        damping=None,
        render_size=96,
        window_size=512,
        render_action=True,
    ):
        self.window_size = ws = window_size
        self.render_size = render_size
        self.render_action = render_action
        self.legacy = legacy

        self.observation_space = spaces.Box(low=0, high=ws, shape=(5,), dtype=np.float64)
        self.action_space = spaces.Box(low=0, high=ws, shape=(2,), dtype=np.float64)

        self.block_cog = block_cog
        self.damping = damping
        self.window = None
        self.clock = None
        self.screen = None
        self.teleop = None
        self.sim_hz = 100
        self.k_p, self.k_v = 100, 20

        self.space = None
        self.agent = None
        self.block = None
        self.goal_pose = None
        self.goal_body = None

        self._seed = None
        self.max_score = 0

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 2**31)
        self._seed = seed
        np.random.seed(seed)

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0 if self.damping is None else self.damping
        self.teleop = False

        # Add walls
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        # Add agent
        self.agent = self._add_circle((256, 400), 15)

        # Add T-block
        block_body, block_shapes = self._add_tee((256, 300), 0)
        self.block = block_body
        self.block_shapes = block_shapes

        # Goal pose
        self.goal_pose = np.array([256, 256, np.pi / 4])
        self.goal_body = self._add_goal_tee((256, 256), np.pi / 4)

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.elasticity = 1.0
        shape.friction = 0.5
        shape.color = pygame.Color("LightGray")
        return shape

    def _add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("RoyalBlue")
        self.space.add(body, shape)
        return body

    def _add_tee(self, position, angle, scale=30, color="LightSlateGray", mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]

        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        body = pymunk.Body(mass, inertia1 + inertia2)
        body.position = position
        body.angle = angle
        body.friction = 1
        shape1 = pymunk.Poly(body, vertices1)
        shape1.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2 = pymunk.Poly(body, vertices2)
        shape2.color = pygame.Color(color)
        shape2.filter = pymunk.ShapeFilter(mask=mask)

        if self.block_cog is not None:
            body.center_of_gravity = self.block_cog

        self.space.add(body, shape1, shape2)
        return body, [shape1, shape2]

    def _add_goal_tee(self, position, angle):
        body, shapes = self._add_tee(
            position, angle, color="LightGreen",
            mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0x1
        )
        for s in shapes:
            s.sensor = True
        return body

    def reset(self):
        self._setup()
        # Randomize initial agent and block positions
        rs = np.random.RandomState(self._seed)
        # Randomize block position and angle
        block_pos = rs.uniform([100, 100], [400, 400])
        block_angle = rs.uniform(0, 2 * np.pi)
        self.block.position = tuple(block_pos)
        self.block.angle = block_angle
        # Randomize agent position
        agent_pos = rs.uniform([50, 50], [450, 450])
        self.agent.position = tuple(agent_pos)
        # Step physics to settle
        for _ in range(10):
            self.space.step(1.0 / self.sim_hz)
        state = self._get_obs()
        self.max_score = 0
        return state

    def _get_obs(self):
        agent_pos = np.array(self.agent.position)
        block_pos = np.array(self.block.position)
        block_angle = self.block.angle % (2 * np.pi)
        return np.array([agent_pos[0], agent_pos[1],
                        block_pos[0], block_pos[1], block_angle])

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_score(self):
        goal_geom = pymunk_to_shapely(self.goal_body, self.goal_body.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block_shapes)
        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        score = np.clip(coverage / 0.95, 0, 1)
        return score

    def step(self, action):
        dt = 1.0 / self.sim_hz
        target = Vec2d(*action)

        for _ in range(int(self.sim_hz / 10)):
            agent_pos = Vec2d(*self.agent.position)
            agent_vel = Vec2d(*self.agent.velocity)
            delta = target - agent_pos
            accel = self.k_p * delta + self.k_v * (Vec2d(0, 0) - agent_vel)
            self.agent.velocity = tuple(agent_vel + accel * dt)
            self.space.step(dt)

        obs = self._get_obs()
        score = self._get_score()
        self.max_score = max(self.max_score, score)
        done = score > 0.95
        return obs, score, done, {"score": score, "max_score": self.max_score}

    def render(self, mode="rgb_array"):
        return np.zeros((self.render_size, self.render_size, 3), dtype=np.uint8)


def evaluate_policy(model, dataset, device, num_episodes=50, seed_start=0):
    """
    Evaluate a trained model in the PushT environment.

    Returns dict with mean_score, max_score, etc.
    """
    env = PushTEnv()
    scores = []
    max_scores = []

    for ep in range(num_episodes):
        env.seed(seed_start + ep * 1000)
        obs = env.reset()
        obs_history = [obs]

        episode_scores = []
        done = False
        step_count = 0
        max_steps = 300

        while not done and step_count < max_steps:
            # Pad observation history if needed
            while len(obs_history) < dataset.obs_horizon:
                obs_history.insert(0, obs_history[0])

            # Get recent observations
            recent_obs = np.array(obs_history[-dataset.obs_horizon:])  # (obs_horizon, 5)

            # Tokenize observations
            obs_tokens = dataset.obs_tokenizer.encode(recent_obs).flatten()
            obs_tokens_t = torch.tensor(obs_tokens, dtype=torch.long, device=device).unsqueeze(0)

            # Predict actions
            model.eval()
            act_tokens = model.predict_actions(obs_tokens_t)  # (1, act_seq_len)

            # Decode actions
            act_tokens_np = act_tokens.cpu().numpy()[0]
            actions = dataset.decode_actions(act_tokens_np)  # (pred_horizon, 2)

            # Execute action horizon steps
            for t in range(min(dataset.action_horizon, len(actions))):
                action = actions[t]
                action = np.clip(action, 0, 512)
                obs, score, done, info = env.step(action)
                obs_history.append(obs)
                episode_scores.append(score)
                step_count += 1
                if done:
                    break

        max_score = max(episode_scores) if episode_scores else 0
        scores.append(episode_scores[-1] if episode_scores else 0)
        max_scores.append(max_score)

    results = {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "mean_max_score": np.mean(max_scores),
        "std_max_score": np.std(max_scores),
        "num_episodes": num_episodes,
    }
    return results


def evaluate_policy_v2(model, dataset, device, num_episodes=50, seed_start=0):
    """
    Evaluate V2 model (continuous obs input) in PushT environment.
    """
    env = PushTEnv()
    scores = []
    max_scores = []

    for ep in range(num_episodes):
        env.seed(seed_start + ep * 1000)
        obs = env.reset()
        obs_history = [obs]

        episode_scores = []
        done = False
        step_count = 0
        max_steps = 300

        while not done and step_count < max_steps:
            while len(obs_history) < dataset.obs_horizon:
                obs_history.insert(0, obs_history[0])

            recent_obs = np.array(obs_history[-dataset.obs_horizon:])  # (obs_horizon, 5)

            # Normalize observations
            obs_norm = dataset.normalize_obs(recent_obs)
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)

            # Predict action tokens
            model.eval()
            act_tokens = model.predict_actions(obs_t)  # (1, act_seq_len)

            # Decode actions
            act_tokens_np = act_tokens.cpu().numpy()[0]
            actions = dataset.decode_actions(act_tokens_np)  # (pred_horizon, 2)

            # Execute action horizon steps
            for t in range(min(dataset.action_horizon, len(actions))):
                action = actions[t]
                action = np.clip(action, 0, 512)
                obs, score, done, info = env.step(action)
                obs_history.append(obs)
                episode_scores.append(score)
                step_count += 1
                if done:
                    break

        max_score = max(episode_scores) if episode_scores else 0
        scores.append(episode_scores[-1] if episode_scores else 0)
        max_scores.append(max_score)

    return {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "mean_max_score": np.mean(max_scores),
        "std_max_score": np.std(max_scores),
        "num_episodes": num_episodes,
    }
