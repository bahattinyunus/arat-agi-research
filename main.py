import asyncio
import numpy as np
import time
import logging
from core_cognition.reasoning_engine import ActiveInferenceAgent, HierarchicalActiveInferenceAgent
from core_cognition.episodic_memory import SemanticMemory, EpisodicMemory, WorkingMemory
from core_cognition.agent_orchestration import EpistemicMessageBroker, EpistemicMessage
from autonomous_control.sensor_fusion import SensorFusionNode, EKFSensorFusionNode
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
    logger.info("Featuring Hierarchical Active Inference & EKF Fusion")
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
    semantic_mem.store("Hierarchical Active Inference operates at goal and motor scales.", {"source": "init"})
    semantic_mem.store("Extended Kalman Filters fuse non-linear velocity and odometry.", {"source": "init"})
    
    # Set up Hierarchical Active Inference Agent
    # High-level params (Goal level)
    high_states = ["Goal_Advance", "Goal_Recon", "Goal_SafeState"]
    high_observations = ["NoThreat", "LiDAR_Anomaly", "LowPower"]
    high_actions = ["ExecuteAdvance", "ExecuteRecon", "ExecuteSafety"]
    
    high_params = {
        "num_states": len(high_states),
        "num_obs": len(high_observations),
        "num_actions": len(high_actions),
        "learning_rate": 0.1
    }
    
    # Low-level params (Reactive control level)
    low_states = ["MovingForward", "SlowingDown", "EmergencyStopped"]
    low_observations = ["ClearPath", "ObstacleDetected", "CollisionImminent"]
    low_actions = ["ApplyAcceleration", "ApplyBrake", "ApplyFullStop"]
    
    low_params = {
        "num_states": len(low_states),
        "num_obs": len(low_observations),
        "num_actions": len(low_actions),
        "learning_rate": 0.2
    }
    
    # Goal to preference mapping matrix (high_actions x low_obs)
    # High-level goal actions set the prior preference C of the low level agent
    goal_to_preference_matrix = np.array([
        [0.90, 0.09, 0.01],  # ExecuteAdvance: strongly prefers ClearPath
        [0.20, 0.70, 0.10],  # ExecuteRecon: prefers ObstacleDetected to investigate
        [0.01, 0.09, 0.90]   # ExecuteSafety: strongly prefers CollisionImminent to stop/caution
    ])
    
    hierarchical_agent = HierarchicalActiveInferenceAgent(
        high_params,
        low_params,
        goal_to_preference_matrix
    )

    # Set prior state matrices for high agent
    hierarchical_agent.high_agent.set_matrices(
        C=np.array([0.8, 0.15, 0.05]), # Prior preferences for high level observations
        D=np.array([0.7, 0.2, 0.1])    # Prior belief about initial goal states
    )
    
    # ==========================================
    # 2. ORKESTRASYON VE HABERLEŞME (Orchestration)
    # ==========================================
    broker = EpistemicMessageBroker()
    
    logger.info("[Orchestrator] Fusing beliefs from disparate specialist agents...")
    agent_a_belief = np.array([0.7, 0.2, 0.1])
    agent_b_belief = np.array([0.5, 0.4, 0.1])
    agent_c_belief = np.array([0.1, 0.8, 0.1])
    beliefs = [agent_a_belief, agent_b_belief, agent_c_belief]
    confidences = [0.9, 0.8, 0.4]
    
    linear_fused = broker.fuse_beliefs(beliefs, confidences)
    log_linear_fused = broker.fuse_beliefs_log_linear(beliefs, confidences)
    
    logger.info(f"[Orchestrator] Linear Fused (Consensus): {linear_fused.round(3).tolist()}")
    logger.info(f"[Orchestrator] Log-Linear Fused (Expert Product): {log_linear_fused.round(3).tolist()}")

    # ==========================================
    # 3. NON-LINEAR SENSOR FUSION & ROBOTICS
    # ==========================================
    dt = 0.1
    # Non-linear EKF Node
    ekf_node = EKFSensorFusionNode(dt=dt, initial_pose=(0.0, 0.0, 0.0))
    
    # PID controller for linear velocity
    pid = PIDController(kp=2.0, ki=0.5, kd=0.1, kff=1.0, output_min=-5.0, output_max=5.0)
    
    # Differential Drive Kinematics
    kinematics = DifferentialDriveKinematics(wheel_radius=0.08, wheel_base=0.4, initial_pose=(0.0, 0.0, 0.0))

    logger.info("==================================================")
    logger.info("Running Real-Time Non-Linear Physics & Hierarchical Control")
    logger.info("==================================================")
    
    # Target linear velocity command
    target_v = 1.5
    target_w = 0.2  # curve motion
    
    # Simulated true state
    true_x = 0.0
    true_y = 0.0
    true_theta = 0.0
    
    for step in range(12):
        # 1. Simulate actual robot physics update using kinematics
        # Determine actual control output from PID
        est_state = ekf_node.get_state_estimate()
        v_est = target_v # mock tracking speed
        w_est = target_w
        
        # Calculate kinematics motor outputs
        omega_l, omega_r = kinematics.inverse_kinematics(v=v_est, omega=w_est)
        # Update pose in kinematics
        kinematics.forward_kinematics(omega_l, omega_r, dt)
        pose = kinematics.get_pose()
        
        true_x = pose["x"]
        true_y = pose["y"]
        true_theta = pose["theta"]
        
        # 2. Simulate sensor measurements with noise
        # GPS/LiDAR absolute position measurements
        noise_x = np.random.normal(0, 0.05)
        noise_y = np.random.normal(0, 0.05)
        sim_pos = (true_x + noise_x, true_y + noise_y) if step % 2 == 0 else None
        
        # 3. Process EKF step
        ekf_est = ekf_node.process_step(linear_vel=v_est, angular_vel=w_est, meas_pos=sim_pos)
        
        # 4. Map simulated states to discrete observations for Hierarchical Active Inference
        # High level observations: depends on steps/mission phases (simulated anomaly/battery levels)
        if step < 4:
            high_obs_idx = 0  # NoThreat
        elif step < 8:
            high_obs_idx = 1  # LiDAR_Anomaly detected
        else:
            high_obs_idx = 2  # LowPower mode
            
        # Low level observations: depends on proximity to a mock obstacle at (x=2.0, y=0.5)
        dist_to_obstacle = np.sqrt((ekf_est["x"] - 2.0)**2 + (ekf_est["y"] - 0.5)**2)
        if dist_to_obstacle < 0.5:
            low_obs_idx = 2   # CollisionImminent
        elif dist_to_obstacle < 1.2:
            low_obs_idx = 1   # ObstacleDetected
        else:
            low_obs_idx = 0   # ClearPath
            
        # 5. Run Hierarchical Active Inference step
        res = hierarchical_agent.run_step(high_obs=high_obs_idx, low_obs=low_obs_idx)
        
        h_action = res["high_action"]
        l_action = res["low_action"]
        h_belief = res["high_belief"]
        l_belief = res["low_belief"]
        
        # Adjust target velocities based on active inference actions
        if low_actions[l_action] == "ApplyFullStop":
            target_v = 0.0
            target_w = 0.0
        elif low_actions[l_action] == "ApplyBrake":
            target_v = 0.5
            target_w = 0.1
        else:
            target_v = 1.5
            target_w = 0.2
            
        # Log to memory
        working_mem.add(f"Pose: ({ekf_est['x']:.2f}, {ekf_est['y']:.2f}) | LowAction: {low_actions[l_action]}", importance=0.85)
        episodic_mem.record_episode(
            state=low_states[np.argmax(l_belief)],
            action=low_actions[l_action],
            observation=low_observations[low_obs_idx],
            reward=1.0 if low_obs_idx == 0 else (0.2 if low_obs_idx == 1 else -2.0)
        )
        
        logger.info(
            f"Step {step:02d} | EKF Pose: ({ekf_est['x']:.2f}, {ekf_est['y']:.2f}, {ekf_est['theta']:.2f}) | "
            f"Goal: {high_actions[h_action]} | "
            f"Reactive Action: {low_actions[l_action]} | Covariance Var: {[round(c, 4) for c in ekf_est['covariance']]}"
        )
        await asyncio.sleep(0.01)

    logger.info("==================================================")
    logger.info("Running Semantic Memory Consolidation Process...")
    logger.info("==================================================")
    
    consolidated_facts = semantic_mem.consolidate_episodes(episodic_mem)
    logger.info(f"Consolidation process complete. Stored {consolidated_facts} new facts.")
    
    # Query semantic memory to verify consolidation
    query_res = semantic_mem.query("What action do we take under obstacles or safety states?", top_k=2)
    for idx, res in enumerate(query_res):
        logger.info(f"Consolidated Fact {idx+1}: '{res['text']}' (Similarity Score: {res['score']:.4f})")

    logger.info("==================================================")
    logger.info("ARAT OMEGA/AGI Integrated Simulation Completed Successfully!")
    logger.info("==================================================")

if __name__ == "__main__":
    asyncio.run(main())
