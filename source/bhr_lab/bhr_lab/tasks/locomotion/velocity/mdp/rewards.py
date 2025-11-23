from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

import isaaclab.envs.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

DEG2RAD = 3.14159265358979323846 / 180.0

def biped_leg_no_cross(
    env: ManagerBasedRLEnv, 
    assetl_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    assetr_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tolerance: float = 2.0,
) -> torch.Tensor:
    """Penalize leg cross through the difference of hip yaw joint positions.
    
    Args:
        env: The environment instance
        assetl_cfg: Configuration for the left leg
        assetr_cfg: Configuration for the right leg
        tolerance: Tolerance in degrees before penalization starts
        
    Returns:
        Reward penalty for leg crossing
    """
    # extract the used quantities (to enable type-hinting)
    assetl: Articulation = env.scene[assetl_cfg.name]
    assetr: Articulation = env.scene[assetr_cfg.name]
    # compute difference, minus is better
    angle = assetr.data.joint_pos[:, assetr_cfg.joint_ids] - assetl.data.joint_pos[:, assetl_cfg.joint_ids]
    angle = torch.clamp(angle, min=tolerance * DEG2RAD)
    return torch.sum(torch.square(angle), dim=1)

def biped_symmetry_air_time(
    env: ManagerBasedRLEnv, 
    l_sensor_cfg: SceneEntityCfg,
    r_sensor_cfg: SceneEntityCfg,
    tolerance: float = 0.05,
    constant_reward: float = 0.01,
) -> torch.Tensor:
    """Penalize asymmetric air time between left and right feet.
    
    Encourages a balanced gait by penalizing differences in how long
    each foot stays in the air.
    
    Args:
        env: The environment instance
        l_sensor_cfg: Configuration for the left foot contact sensor
        r_sensor_cfg: Configuration for the right foot contact sensor
        tolerance: Threshold for time difference before penalization
        constant_reward: Reward value when within tolerance
        
    Returns:
        Reward penalty for asymmetric air time
    """
    # extract the used quantities (to enable type-hinting)
    l_contact_sensor: ContactSensor = env.scene.sensors[l_sensor_cfg.name]
    r_contact_sensor: ContactSensor = env.scene.sensors[r_sensor_cfg.name]
    # compute the reward
    l_last_air_time = l_contact_sensor.data.last_air_time[:, l_sensor_cfg.body_ids]
    r_last_air_time = r_contact_sensor.data.last_air_time[:, r_sensor_cfg.body_ids]
    diff = torch.sum(torch.abs(l_last_air_time - r_last_air_time), dim=1)
    reward = torch.where(diff > tolerance, diff, torch.zeros_like(diff) - constant_reward)
    return reward

def biped_symmetry_contact_time(
    env: ManagerBasedRLEnv, 
    l_sensor_cfg: SceneEntityCfg,
    r_sensor_cfg: SceneEntityCfg,
    tolerance: float = 0.05,
    constant_reward: float = 0.01,
) -> torch.Tensor:
    """Penalize asymmetric ground contact time between left and right feet.
    
    Encourages a balanced gait by penalizing differences in how long
    each foot stays in contact with the ground.
    
    Args:
        env: The environment instance
        l_sensor_cfg: Configuration for the left foot contact sensor
        r_sensor_cfg: Configuration for the right foot contact sensor
        tolerance: Threshold for time difference before penalization
        constant_reward: Reward value when within tolerance
        
    Returns:
        Reward penalty for asymmetric contact time
    """
    # extract the used quantities (to enable type-hinting)
    l_contact_sensor: ContactSensor = env.scene.sensors[l_sensor_cfg.name]
    r_contact_sensor: ContactSensor = env.scene.sensors[r_sensor_cfg.name]
    # compute the reward
    l_last_contact_time = l_contact_sensor.data.last_contact_time[:, l_sensor_cfg.body_ids]
    r_last_contact_time = r_contact_sensor.data.last_contact_time[:, r_sensor_cfg.body_ids]
    diff = torch.sum(torch.abs(l_last_contact_time - r_last_contact_time), dim=1)
    reward = torch.where(diff > tolerance, diff, torch.zeros_like(diff) - constant_reward)
    return reward

def joint_deviation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default pose using L2 norm.
    
    Encourages the robot to maintain a natural posture by penalizing
    deviations from the default joint positions.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot
        
    Returns:
        Reward penalty for joint deviations
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(angle), dim=1)

def biped_no_double_feet_air(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward having only one foot in the air at a time.
    
    This function penalizes having both feet in the air simultaneously,
    encouraging a stable walking gait where at least one foot is always
    in contact with the ground.
    
    Args:
        env: The environment instance
        sensor_cfg: Configuration for the foot contact sensors
        
    Returns:
        Reward for appropriate foot placement
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    double_air = torch.sum(in_contact.int(), dim=1) == 0
    reward = torch.min(torch.where(double_air.unsqueeze(-1), air_time, 0.0), dim=1)[0]
    return reward

def joint_coordination(
    env: ManagerBasedRLEnv, 
    joint1_cfg: SceneEntityCfg, 
    joint2_cfg: SceneEntityCfg,
    tolerance: float = 5.0,  # deg
    ratio: float = 1.0,
    coordination_reward: float = 0.05
) -> torch.Tensor:
    """Penalize lack of coordination between two joints.
    
    This function enforces a desired relationship between two joints,
    which is useful for enforcing natural movement patterns like 
    coordinated shoulder-hip motion during walking.
    
    Args:
        env: The environment instance
        joint1_cfg: Configuration for the first joint
        joint2_cfg: Configuration for the second joint
        tolerance: Angle tolerance in degrees before penalization
        ratio: Desired ratio between joint1 and joint2 angles
        coordination_reward: Reward value for staying within tolerance
        
    Returns:
        Reward for joint coordination
    """
    # extract the used quantities (to enable type-hinting)
    joint1: Articulation = env.scene[joint1_cfg.name]
    joint2: Articulation = env.scene[joint2_cfg.name]
    # compute devirations
    angle = torch.abs(joint1.data.joint_pos[:, joint1_cfg.joint_ids] - joint1.data.default_joint_pos[:, joint1_cfg.joint_ids] - ratio * (joint2.data.joint_pos[:, joint2_cfg.joint_ids] - joint2.data.default_joint_pos[:, joint2_cfg.joint_ids]))
    angle = torch.where(angle > tolerance * DEG2RAD, angle, torch.zeros_like(angle) - coordination_reward)
    reward = torch.max(angle, dim=1).values
    return reward

def distance_in_root(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    min_dist: float, 
    max_dist: float, 
    axis_index: int = 1,
    constant_reward: float = 0.1
) -> torch.Tensor:
    """Penalize inappropriate lateral distance between feet.
    
    This function encourages the robot to maintain an appropriate stance width
    by penalizing when feet are too close together (unstable) or too far apart (unnatural).
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot
        min_dist: Minimum allowed distance between feet
        max_dist: Maximum allowed distance between feet
        axis_index: Index of the axis to measure distance (default is 1 for lateral distance)
        constant_reward: Reward value when within distance bounds
        
    Returns:
        Reward for appropriate lateral foot distance
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    root_quat = asset.data.root_quat_w
    asset_pos_world = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    # asset_pos_body = quat_rotate_inverse(yaw_quat(root_quat.unsqueeze(1)), asset_pos_world - root_pos.unsqueeze(1))
    yaw_quat_only = yaw_quat(root_quat)
    yaw_quat_expanded = yaw_quat_only.unsqueeze(1).expand(-1, asset_pos_world.shape[1], -1)
    asset_pos_body = quat_apply_inverse(yaw_quat_expanded, asset_pos_world - root_pos.unsqueeze(1))
    asset_dist = torch.abs(asset_pos_body[:, 0, axis_index] - asset_pos_body[:, 1, axis_index]).unsqueeze(1)

    dist = torch.where(
        asset_dist < min_dist, 
        torch.abs(asset_dist - min_dist), 
        torch.where(
            asset_dist > max_dist, 
            torch.abs(asset_dist - max_dist), 
            torch.zeros_like(asset_dist) - constant_reward
        )
    )
    reward = torch.min(dist, dim=1).values
    return reward

def lateral_distance(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    min_dist: float, 
    max_dist: float, 
    constant_reward: float = 0.1
) -> torch.Tensor:
    return distance_in_root(env, asset_cfg, min_dist, max_dist, axis_index=1, constant_reward=constant_reward)

def longitudinal_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_dist: float,
    max_dist: float,
    constant_reward: float = 0.1
) -> torch.Tensor:
    return distance_in_root(env, asset_cfg, min_dist, max_dist, axis_index=0, constant_reward=constant_reward)

def double_feet_in_air_cmd(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    threshold: float, 
    sensor_cfg: SceneEntityCfg, 
    vel_threshold: float = 0.1
) -> torch.Tensor:
    """Reward having both feet in the air simultaneously for jumping.
    
    This function rewards the robot for keeping both feet in the air,
    which is crucial for jumping maneuvers. Only applies when velocity
    commands exceed the threshold.
    
    Args:
        env: The environment instance
        command_name: Name of the velocity command
        threshold: Maximum reward value
        sensor_cfg: Configuration for the foot contact sensors
        vel_threshold: Minimum velocity command to apply reward
        
    Returns:
        Reward for proper jumping behavior
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    both_feet_in_air = torch.sum(in_contact.int(), dim=1) == 0
    reward = torch.min(torch.where(both_feet_in_air.unsqueeze(-1), air_time, torch.zeros_like(air_time)), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > vel_threshold
    return reward

def double_feet_in_contact_cmd(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    threshold: float, 
    sensor_cfg: SceneEntityCfg, 
    vel_threshold: float = 0.1
) -> torch.Tensor:
    """Reward having both feet in contact with the ground simultaneously.
    
    This function encourages stable stance by rewarding when both feet
    are in contact with the ground. Only applies when velocity
    commands exceed the threshold.
    
    Args:
        env: The environment instance
        command_name: Name of the velocity command
        threshold: Maximum reward value
        sensor_cfg: Configuration for the foot contact sensors
        vel_threshold: Minimum velocity command to apply reward
        
    Returns:
        Reward for double support stance
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    both_feet_in_contact = torch.sum(in_contact.int(), dim=1) == 2
    reward = torch.min(torch.where(both_feet_in_contact.unsqueeze(-1), contact_time, torch.zeros_like(contact_time)), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > vel_threshold
    return reward

def single_feet_in_air_cmd(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    reward_threshold: float, 
    sensor_cfg: SceneEntityCfg, 
    vel_threshold: float = 0.1
) -> torch.Tensor:
    """Reward having only one foot in the air at a time for walking.
    
    This function encourages a natural walking gait by rewarding when
    exactly one foot is in the air at a time. Only applies when velocity
    commands exceed the threshold.
    
    Args:
        env: The environment instance
        command_name: Name of the velocity command
        reward_threshold: Maximum reward value
        sensor_cfg: Configuration for the foot contact sensors
        vel_threshold: Minimum velocity command to apply reward
        
    Returns:
        Reward for single support stance during walking
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=reward_threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > vel_threshold
    return reward

def specific_single_air_cmd(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    reward_threshold: float, 
    specific_sensor_cfg: SceneEntityCfg,
    other_sensor_cfg: SceneEntityCfg,
    vel_threshold: float = 0.1
) -> torch.Tensor:
    """Reward having a specific link in the air while other links maintain ground contact.
    
    This function can be used to enforce specific gait patterns by rewarding
    when a designated link is in the air while others stay grounded.
    
    Args:
        env: The environment instance
        command_name: Name of the velocity command
        reward_threshold: Maximum reward value
        specific_sensor_cfg: Configuration for the sensor on the link that should be in air
        other_sensor_cfg: Configuration for the sensor on the link that should be in contact
        vel_threshold: Minimum velocity command to apply reward
        
    Returns:
        Reward for the specific foot placement pattern
    """
    specific_contact_sensor: ContactSensor = env.scene.sensors[specific_sensor_cfg.name]
    other_contact_sensor: ContactSensor = env.scene.sensors[other_sensor_cfg.name]
    # compute the reward
    specific_air_time = specific_contact_sensor.data.current_air_time[:, specific_sensor_cfg.body_ids]
    other_contact_time = other_contact_sensor.data.current_contact_time[:, other_sensor_cfg.body_ids]
    specific_in_air = specific_air_time > 0.0
    other_in_contact = other_contact_time > 0.0
    specific_air = torch.sum(specific_in_air.int(), dim=1) == 1
    others_contact = torch.sum(other_in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(specific_air.unsqueeze(-1) & others_contact.unsqueeze(-1), specific_air_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=reward_threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > vel_threshold
    return reward

def desired_step_time_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    desired_time: float,
    tolerance: float = 0.1,
    vel_threshold: float = 0.1,
    reward_value: float = 0.1
) -> torch.Tensor:
    """Reward the agent for taking steps with a specific timing.
    
    This function encourages a regular, rhythmic gait by rewarding steps
    that occur within a desired time frame. Only applies when velocity
    commands exceed the threshold.
    
    Args:
        env: The environment instance
        command_name: Name of the velocity command
        sensor_cfg: Configuration for the foot contact sensors
        desired_time: Target step duration in seconds
        tolerance: Time tolerance before penalization starts
        vel_threshold: Minimum velocity command to apply reward
        reward_value: Reward magnitude for correct timing
        
    Returns:
        Reward for maintaining the desired step timing
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    step_time = (air_time + contact_time) / 2.0
    deviation = torch.abs(step_time - desired_time)
    reward = torch.where(deviation < tolerance, torch.zeros_like(deviation) - reward_value, deviation)
    reward = torch.sum(reward, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > vel_threshold
    return reward

def saggital_orientation_l2(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize non-flat base orientation in the sagittal plane (forward-backward tilt).
    
    This function encourages the robot to maintain an upright posture by
    penalizing forward or backward tilting.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot
        
    Returns:
        Reward penalty for sagittal plane tilt
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 0])

def coronal_orientation_l2(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize non-flat base orientation in the coronal plane (side-to-side tilt).
    
    This function encourages the robot to maintain an upright posture by
    penalizing leftward or rightward tilting.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot
        
    Returns:
        Reward penalty for coronal plane tilt
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 1])

def link_keep_flat(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize non-flat orientation of specified links.
    
    This function encourages specific links (like the pelvis or torso)
    to remain horizontal by penalizing deviations from a flat orientation.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration specifying which links to keep flat
        
    Returns:
        Reward penalty for non-flat link orientation
    """
    asset: Articulation = env.scene[asset_cfg.name]
    link_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    gravity_vec_w = asset.data.GRAVITY_VEC_W  # (num_robots, 3)

    # Broadcast gravity_vec_w to (num_robots, num_bodies, 3)
    gravity_vec_w_expanded = gravity_vec_w.unsqueeze(1).expand(-1, link_quat_w.shape[1], -1)
    
    # Calculate inverse-rotated gravity vector (num_robots, num_bodies, 3)
    gravity_link = quat_apply_inverse(link_quat_w, gravity_vec_w_expanded)
    
    # Calculate the square sum of horizontal components (x, y) (num_robots, num_bodies)
    reward_per_body = torch.sum(torch.square(gravity_link[:, :, :2]), dim=2)
    
    # Take the maximum value across all bodies as the final reward (num_robots,)
    reward = torch.max(reward_per_body, dim=1).values

    return reward

def link_ang_vel_l2(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize excessive angular velocity of specified links.
    
    This function encourages smooth, controlled movement by penalizing
    rapid rotational motions of designated links.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration specifying which links to monitor
        
    Returns:
        Reward penalty for high angular velocity
    """
    asset: Articulation = env.scene[asset_cfg.name]
    link_ang_vel = quat_apply_inverse(asset.data.body_link_quat_w, asset.data.body_link_ang_vel_w)
    return torch.sum(torch.square(link_ang_vel[:, :, :2]), dim=2).max(dim=1).values

def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from default pose when no movement command is given.
    
    This function encourages the robot to maintain a default standing pose
    when no movement is requested, preventing unnecessary movement.
    
    Args:
        env: The environment instance
        command_name: Name of the velocity command
        command_threshold: Velocity threshold below which to apply penalty
        asset_cfg: Configuration for the robot
        
    Returns:
        Reward penalty for unnecessary movement
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) < command_threshold
    return reward

def desired_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    std: float
) -> torch.Tensor:
    """Reward low contact forces on specified links.
    
    This function encourages gentle interactions with the environment
    by rewarding when contact forces remain below a specified threshold.
    
    Args:
        env: The environment instance
        sensor_cfg: Configuration for the contact force sensors
        threshold: Maximum contact force to avoid penalty
        std: Standard deviation for reward scaling

    Returns:
        Reward for maintaining low contact forces
    """
    violations = mdp.contact_forces(env, threshold, sensor_cfg)
    return torch.exp(-0.5 * (violations / std) ** 2)

def contact_forces_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize high contact forces on specified links using L2 squared kernel.
    
    This function encourages gentle interactions with the environment
    by penalizing large contact forces using the L2 squared penalty.
    
    Args:
        env: The environment instance
        sensor_cfg: Configuration for the contact force sensors

    Returns:
        Reward penalty for high contact forces (L2 squared sum)
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # get current net contact forces for the specified body IDs
    net_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    # compute L2 squared penalty: sum of squared force magnitudes
    force_magnitudes = torch.norm(net_contact_forces, dim=-1)  # compute magnitude for each contact
    return torch.sum(torch.square(force_magnitudes), dim=1)  # L2 squared penalty