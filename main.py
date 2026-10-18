import asyncio
import numpy as np
import time
import logging
from core_cognition.reasoning_engine import ActiveInferenceAgent, HierarchicalActiveInferenceAgent
from core_cognition.episodic_memory import SemanticMemory, EpisodicMemory, WorkingMemory
from core_cognition.agent_orchestration import EpistemicMessageBroker, EpistemicMessage
from core_cognition.quantum_cognition import QuantumDecisionAgent
from core_cognition.constructor_theory import PhysicalState, PhysicalTask, Constructor, ConstructorTaskPlanner
from autonomous_control.sensor_fusion import EKFSensorFusionNode, UKFSensorFusionNode
from autonomous_control.kinematic_loops import PIDController, DifferentialDriveKinematics
from autonomous_control.trajectory_planning import DynamicWindowApproachPlanner, DWAPlannerConfig, TrajectoryGenerator2D

# Set up logging formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("OMEGA-AGI-Master")

async def main():
    logger.info("======================================================================")
    logger.info("  ARAT OMEGA/AGI v1.2 MASTER INTEGRATED SIMULATION")
    logger.info("  Quantum Cognition | Constructor Theory | Hierarchical FEP | UKF & DWA")
    logger.info("======================================================================")

    # =========================================================================
    # PHASE 1: CONSTRUCTOR THEORY TASK FEASIBILITY VERIFICATION
    # =========================================================================
    logger.info("\n--- [Phase 1: Constructor Theory Task Network Validation] ---")
    c_robot = Constructor("AratEdgeRobot", max_power=75.0, degradation_rate=0.01)
    c_robot.register_capability("NavigateSector")
    c_robot.register_capability("SensorScan")

    task_planner = ConstructorTaskPlanner([c_robot])

    state_depot = PhysicalState("DepotBase", {"loaded": False}, energy=100.0, entropy=1.0)
    state_recon = PhysicalState("WaypointRecon", {"scanned": True}, energy=85.0, entropy=1.2)
    state_target = PhysicalState("MissionObjective", {"target_acquired": True}, energy=65.0, entropy=1.4)

    task_nav = PhysicalTask("NavigateSector", state_depot, state_recon, required_work=15.0)
    task_scan = PhysicalTask("SensorScan", state_recon, state_target, required_work=20.0)

    feasibility = task_planner.verify_plan([task_nav, task_scan])
    logger.info(f"Constructor Plan Feasibility: {feasibility['all_performable']} (Verified {feasibility['total_tasks']} physical tasks)")
    for report in feasibility["step_reports"]:
        logger.info(f"  Task: '{report['task']}' -> Performable: {report['performable']} | Assigned Constructor: {report.get('constructor', 'None')}")

    # =========================================================================
    # PHASE 2: QUANTUM COGNITION DELIBERATION UNDER AMBIGUITY
    # =========================================================================
    logger.info("\n--- [Phase 2: Quantum Cognitive Deliberation & Collapse] ---")
    cognitive_hypotheses = ["ExploreAutonomously", "InspectObstacle", "ConserveEnergy"]
    quantum_agent = QuantumDecisionAgent(
        state_labels=cognitive_hypotheses,
        oscillation_freq=1.5,
        decoherence_rate=0.08
    )
    
    # Evolve cognitive mind via Lindblad dynamics (unifying deliberate thought with sensory noise)
    quantum_agent.inject_evidence_potential(state_idx=0, strength=1.2)
    quantum_agent.evolve(dt=0.05, steps=10)
    
    pre_measurement_probs = quantum_agent.get_epistemic_belief()
    logger.info(f"Quantum Superposition Amplitudes: {[round(float(p), 3) for p in pre_measurement_probs]}")
    
    # Born rule projection measurement
    decision = quantum_agent.project_decision()
    logger.info(f"Post-Measurement State Collapsed into: '{decision['decision_label']}' (Purity: {decision['purity']:.3f}, Entropy: {decision['entropy']:.3f})")

    # =========================================================================
    # PHASE 3: BİLİŞSEL VE BELLEK MODÜLLERİ (Cognition & Hierarchical Active Inference)
    # =========================================================================
    logger.info("\n--- [Phase 3: Hierarchical Active Inference Setup] ---")
    working_mem = WorkingMemory(capacity=6, decay_rate=0.1)
    episodic_mem = EpisodicMemory()
    semantic_mem = SemanticMemory(dimension=128)

    semantic_mem.store("ARAT Edge runtimes leverage deterministic Arena allocators and quantized kernels.", {"source": "init"})
    semantic_mem.store("Constructor Theory determines performable transformations before motor commitment.", {"source": "init"})
    semantic_mem.store("Quantum Cognition captures contextual order effects and belief superposition.", {"source": "init"})

    high_states = ["Goal_Advance", "Goal_Recon", "Goal_SafeState"]
    high_observations = ["NoThreat", "LiDAR_Anomaly", "LowPower"]
    high_actions = ["ExecuteAdvance", "ExecuteRecon", "ExecuteSafety"]

    high_params = {
        "num_states": len(high_states),
        "num_obs": len(high_observations),
        "num_actions": len(high_actions),
        "learning_rate": 0.1
    }

    low_states = ["MovingForward", "SlowingDown", "EmergencyStopped"]
    low_observations = ["ClearPath", "ObstacleDetected", "CollisionImminent"]
    low_actions = ["ApplyAcceleration", "ApplyBrake", "ApplyFullStop"]

    low_params = {
        "num_states": len(low_states),
        "num_obs": len(low_observations),
        "num_actions": len(low_actions),
        "learning_rate": 0.2
    }

    goal_to_preference_matrix = np.array([
        [0.90, 0.09, 0.01],
        [0.20, 0.70, 0.10],
        [0.01, 0.09, 0.90]
    ])

    hierarchical_agent = HierarchicalActiveInferenceAgent(
        high_params,
        low_params,
        goal_to_preference_matrix
    )

    # Set prior state matrices for high agent
    hierarchical_agent.high_agent.set_matrices(
        C=np.array([0.8, 0.15, 0.05]),
        D=np.array([0.7, 0.2, 0.1])
    )

    # =========================================================================
    # PHASE 4: MULTI-AGENT EPISTEMIC CONSENSUS BROKER
    # =========================================================================
    logger.info("\n--- [Phase 4: Epistemic Multi-Agent Opinion Pooling] ---")
    broker = EpistemicMessageBroker()
    agent_a_belief = np.array([0.75, 0.20, 0.05])
    agent_b_belief = np.array([0.60, 0.30, 0.10])
    agent_c_belief = np.array([0.30, 0.60, 0.10])
    beliefs = [agent_a_belief, agent_b_belief, agent_c_belief]
    confidences = [0.95, 0.85, 0.50]

    linear_fused = broker.fuse_beliefs(beliefs, confidences)
    log_linear_fused = broker.fuse_beliefs_log_linear(beliefs, confidences)

    logger.info(f"Consensus Pool (Linear):     {linear_fused.round(3).tolist()}")
    logger.info(f"Expert Product (Log-Linear): {log_linear_fused.round(3).tolist()}")

    # =========================================================================
    # PHASE 5: DUAL NON-LINEAR SENSOR FUSION (EKF & UKF) & LOCAL DWA PLANNING
    # =========================================================================
    logger.info("\n--- [Phase 5: Real-Time Dual EKF/UKF Estimation & DWA Navigation Loop] ---")
    dt = 0.1
    ekf_node = EKFSensorFusionNode(dt=dt, initial_pose=(0.0, 0.0, 0.0))
    ukf_node = UKFSensorFusionNode(dt=dt, initial_pose=(0.0, 0.0, 0.0))
    kinematics = DifferentialDriveKinematics(wheel_radius=0.08, wheel_base=0.4, initial_pose=(0.0, 0.0, 0.0))
    
    dwa_planner = DynamicWindowApproachPlanner(DWAPlannerConfig(max_speed=1.6, max_yaw_rate=1.2, dt=dt))
    mock_obstacles = [(2.0, 0.3), (2.1, 0.4)]
    goal_coord = (3.5, 0.5)

    current_v = 1.0
    current_w = 0.0

    for step in range(12):
        # 1. Update true kinematics
        omega_l, omega_r = kinematics.inverse_kinematics(v=current_v, omega=current_w)
        kinematics.forward_kinematics(omega_l, omega_r, dt)
        true_pose = kinematics.get_pose()

        # 2. Add simulated measurement noise
        meas = (true_pose["x"] + np.random.normal(0, 0.04), true_pose["y"] + np.random.normal(0, 0.04)) if step % 2 == 0 else None

        # 3. Dual non-linear filter estimation
        ekf_est = ekf_node.process_step(linear_vel=current_v, angular_vel=current_w, meas_pos=meas)
        ukf_est = ukf_node.process_step(linear_vel=current_v, angular_vel=current_w, meas_pos=meas)

        # 4. Map observations to Hierarchical Active Inference
        high_obs_idx = 0 if step < 5 else (1 if step < 9 else 2)
        dist_to_obs = np.sqrt((ukf_est["x"] - mock_obstacles[0][0])**2 + (ukf_est["y"] - mock_obstacles[0][1])**2)
        
        if dist_to_obs < 0.6:
            low_obs_idx = 2  # CollisionImminent
        elif dist_to_obs < 1.3:
            low_obs_idx = 1  # ObstacleDetected
        else:
            low_obs_idx = 0  # ClearPath

        # 5. Run Hierarchical Active Inference step
        fep_res = hierarchical_agent.run_step(high_obs=high_obs_idx, low_obs=low_obs_idx)
        h_action = fep_res["high_action"]
        l_action = fep_res["low_action"]

        # 6. DWA local motion planning
        planned_v, planned_w, _ = dwa_planner.plan(
            current_pose=(ukf_est["x"], ukf_est["y"], ukf_est["theta"]),
            current_v=current_v,
            current_w=current_w,
            goal=goal_coord,
            obstacles=mock_obstacles
        )

        # Active Inference override if emergency stop is triggered
        if low_actions[l_action] == "ApplyFullStop":
            current_v, current_w = 0.0, 0.0
        elif low_actions[l_action] == "ApplyBrake":
            current_v, current_w = min(planned_v, 0.4), planned_w
        else:
            current_v, current_w = planned_v, planned_w

        # Record to memory
        working_mem.add(f"Pose: ({ukf_est['x']:.2f}, {ukf_est['y']:.2f}) | Action: {low_actions[l_action]}", importance=0.85)
        episodic_mem.record_episode(
            state=low_states[np.argmax(fep_res["low_belief"])],
            action=low_actions[l_action],
            observation=low_observations[low_obs_idx],
            reward=1.0 if low_obs_idx == 0 else (0.2 if low_obs_idx == 1 else -2.0)
        )

        logger.info(
            f"Step {step:02d} | UKF Pose: ({ukf_est['x']:.2f}, {ukf_est['y']:.2f}, {ukf_est['theta']:.2f}) | "
            f"EKF Pose: ({ekf_est['x']:.2f}, {ekf_est['y']:.2f}) | "
            f"Goal: {high_actions[h_action]} | Action: {low_actions[l_action]} | Cmd: [v={current_v:.2f}, w={current_w:.2f}]"
        )
        await asyncio.sleep(0.01)

    # =========================================================================
    # PHASE 6: SEMANTIC MEMORY CONSOLIDATION
    # =========================================================================
    logger.info("\n--- [Phase 6: Semantic Memory Consolidation Process] ---")
    consolidated_facts = semantic_mem.consolidate_episodes(episodic_mem)
    logger.info(f"Consolidation complete. Stored {consolidated_facts} generalized facts.")

    query_res = semantic_mem.query("What action do we take under obstacles or safety states?", top_k=2)
    for idx, res in enumerate(query_res):
        logger.info(f"Consolidated Memory Fact {idx+1}: '{res['text']}' (Relevance: {res['score']:.4f})")

    logger.info("\n======================================================================")
    logger.info("  ARAT OMEGA/AGI MASTER SIMULATION COMPLETED WITH FULL SUCCESS!")
    logger.info("======================================================================")

if __name__ == "__main__":
    asyncio.run(main())
