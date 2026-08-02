import asyncio
import numpy as np
import time
import logging
from core_cognition.reasoning_engine import ActiveInferenceAgent
from core_cognition.episodic_memory import SemanticMemory, EpisodicMemory, WorkingMemory
from core_cognition.agent_orchestration import EpistemicMessageBroker, EpistemicMessage
from autonomous_control.sensor_fusion import SensorFusionNode
from autonomous_control.kinematic_loops import PIDController, DifferentialDriveKinematics

# Set up logging formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("OMEGA-AGI-Simulation")

async def main():
    logger.info("==================================================")
    logger.info("Initializing ARAT OMEGA/AGI Integrated Simulation")
    logger.info("==================================================")

    # ==========================================
    # 1. BİLİŞSEL VE BELLEK MODÜLLERİ (Cognition & Memory)
    # ==========================================
    # Initialize Memory
    working_mem = WorkingMemory(capacity=5, decay_rate=0.1)
    episodic_mem = EpisodicMemory()
    semantic_mem = SemanticMemory(dimension=128)
    
    # Store initial knowledge in Semantic Memory
    semantic_mem.store("ARAT Technology & Systems specializes in embedded Edge AI runtimes.", {"source": "init"})
    semantic_mem.store("Active Inference agents minimize free energy to resolve uncertainty.", {"source": "init"})
    semantic_mem.store("Extended Kalman Filters fuse high-frequency IMU and LiDAR data.", {"source": "init"})
    
    # Active Inference Agent (3 states, 3 observations, 3 actions)
    # States: 0 = Safe, 1 = Caution, 2 = Danger
    # Observations: 0 = ClearPath, 1 = ObstacleDetected, 2 = CollisionImminent
    # Actions: 0 = MoveForward, 1 = SlowDown, 2 = EmergencyBrake
    states = ["Safe", "Caution", "Danger"]
    observations = ["ClearPath", "ObstacleDetected", "CollisionImminent"]
    actions = ["MoveForward", "SlowDown", "EmergencyBrake"]
    
    agent = ActiveInferenceAgent(num_states=3, num_obs=3, num_actions=3, learning_rate=0.15)
    
    # Prior state preferences: strong preference for ClearPath (obs 0), aversion to CollisionImminent (obs 2)
    agent.set_matrices(C=np.array([0.9, 0.09, 0.01]))
    
    # ==========================================
    # 2. ORKESTRASYON VE HABERLEŞME (Orchestration)
    # ==========================================
    broker = EpistemicMessageBroker()
    
    # Simulate multi-agent belief fusion
    logger.info("[Orchestrator] Fusing beliefs from disparate specialist agents...")
    agent_a_belief = np.array([0.7, 0.2, 0.1]) # Navigation specialist: highly confident safe
    agent_b_belief = np.array([0.5, 0.4, 0.1]) # LiDAR specialist: moderate caution
    agent_c_belief = np.array([0.1, 0.8, 0.1]) # Safety monitor: sees danger/caution
    
    beliefs = [agent_a_belief, agent_b_belief, agent_c_belief]
    confidences = [0.9, 0.8, 0.4] # Navigation is highly trusted
    
    linear_fused = broker.fuse_beliefs(beliefs, confidences)
    log_linear_fused = broker.fuse_beliefs_log_linear(beliefs, confidences)
    
    logger.info(f"[Orchestrator] Linear Opinion Pool (Mixture): {linear_fused.round(3).tolist()}")
    logger.info(f"[Orchestrator] Log-Linear Pool (Product of Experts): {log_linear_fused.round(3).tolist()}")

    # ==========================================
    # 3. KONTROL VE FÜZYON DÖNGÜSÜ (Control & Fusion)
    # ==========================================
    # Sensor Fusion Node (Kalman Filter)
    dt = 0.1
    fusion_node = SensorFusionNode(dt=dt)
    
    # PID controller for velocity (Kp=1.8, Ki=0.4, Kd=0.1, Kff=1.0)
    pid = PIDController(kp=1.8, ki=0.4, kd=0.1, kff=1.0, output_min=-10.0, output_max=10.0)
    
    # Differential Drive Kinematics
    kinematics = DifferentialDriveKinematics(wheel_radius=0.08, wheel_base=0.4)

    logger.info("==================================================")
    logger.info("Running Real-Time Physics and Control Simulation")
    logger.info("==================================================")
    
    # Target path: straight line movement at target speed
    target_velocity = 2.0  # m/s target
    current_time = 0.0
    
    # True robot states for physics loop
    true_position = 0.0
    true_velocity = 0.0
    
    for step in range(10):
        # 1. Physics Step & Sensor Simulation
        # Simulate constant acceleration to reach target
        accel = target_velocity - true_velocity
        # Add sensor noise
        imu_noise = np.random.normal(0, 0.02)
        lidar_noise = np.random.normal(0, 0.05)
        
        simulated_imu_accel = accel + imu_noise
        simulated_lidar_pos = true_position + true_velocity * dt + lidar_noise if step % 2 == 0 else None
        
        # 2. Kalman Sensor Fusion
        state_est = fusion_node.process_step(simulated_imu_accel, simulated_lidar_pos)
        est_pos = state_est["position"]
        est_vel = state_est["velocity"]
        
        # 3. PID Control with Feedforward
        # Calculate motor acceleration/force command based on target velocity
        control_cmd = pid.compute(setpoint=target_velocity, measurement=est_vel, dt=dt, feedforward_input=0.5)
        
        # 4. Kinematics Updates
        # Calculate wheel velocities
        omega_l, omega_r = kinematics.inverse_kinematics(v=est_vel, omega=0.0)
        # Update pose in map frame
        kinematics.forward_kinematics(omega_l, omega_r, dt)
        pose = kinematics.get_pose()
        
        # Physics update
        true_velocity += control_cmd * dt
        true_position += true_velocity * dt
        
        # 5. Cognitive Loop Step
        # Maps simulated states to discrete observations for Active Inference
        if est_pos > 5.0:
            obs_idx = 2  # CollisionImminent / danger zone
        elif est_pos > 2.0:
            obs_idx = 1  # ObstacleDetected / caution
        else:
            obs_idx = 0  # ClearPath
            
        # Infer state, calculate surprise (VFE), select action
        q_belief = agent.infer_states(obs_idx)
        vfe = agent.calculate_vfe(obs_idx)
        action_idx, _ = agent.select_action()
        
        # Run Dirichlet online likelihood learning
        agent.update_model(obs_idx)
        
        # Log to working and episodic memory
        working_mem.add(f"Position: {est_pos:.2f} | Action: {actions[action_idx]}", importance=0.9)
        episodic_mem.record_episode(
            state=states[np.argmax(q_belief)],
            action=actions[action_idx],
            observation=observations[obs_idx],
            reward=1.0 if obs_idx == 0 else (0.1 if obs_idx == 1 else -2.0)
        )
        
        logger.info(
            f"Step {step} | Pos: {est_pos:.2f}m | Vel: {est_vel:.2f}m/s | "
            f"Obs: {observations[obs_idx]} -> Belief: {states[np.argmax(q_belief)]} | "
            f"Action: {actions[action_idx]} | VFE: {vfe:.4f}"
        )
        await asyncio.sleep(0.02)

    logger.info("==================================================")
    logger.info("Running Semantic Memory Consolidation Process...")
    logger.info("==================================================")
    
    # Consolidate raw episodic sequences into generalized semantic facts
    consolidated_facts = semantic_mem.consolidate_episodes(episodic_mem)
    logger.info(f"Consolidation process complete. Stored {consolidated_facts} new facts.")
    
    # Query semantic memory to verify consolidation
    query_res = semantic_mem.query("What action do we take under Danger or Obstacles?", top_k=2)
    for idx, res in enumerate(query_res):
        logger.info(f"Consolidated Rule {idx+1}: '{res['text']}' (Similarity Score: {res['score']:.4f})")

    logger.info("==================================================")
    logger.info("ARAT OMEGA/AGI Integrated Simulation Completed Successfully!")
    logger.info("==================================================")

if __name__ == "__main__":
    asyncio.run(main())
