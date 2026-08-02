import numpy as np
import json
import logging
from .active_inference import ActiveInferenceAgent

class LLMActiveInferenceReasoner:
    """
    Bridges the mathematical ActiveInferenceAgent with natural language LLM reasoning.
    Translates state spaces into semantic templates, calculates cognitive surprise,
    and runs the active inference loop using LLM observations.
    """
    def __init__(self, states_desc, observations_desc, actions_desc, llm_client=None):
        self.states_desc = states_desc          # List of strings describing states
        self.observations_desc = observations_desc  # List of strings describing observations
        self.actions_desc = actions_desc        # List of strings describing actions
        
        self.num_states = len(states_desc)
        self.num_obs = len(observations_desc)
        self.num_actions = len(actions_desc)
        
        self.agent = ActiveInferenceAgent(self.num_states, self.num_obs, self.num_actions)
        self.llm_client = llm_client
        self.logger = logging.getLogger("LLMReasoner")
        logging.basicConfig(level=logging.INFO)

        # Set up default transition beliefs
        # Action index maps to target transitions. Let's initialize B and A matrixes.
        A = np.eye(self.num_obs, self.num_states) # Default identity mapping
        if self.num_obs != self.num_states:
            A = np.ones((self.num_obs, self.num_states)) / self.num_obs
        
        # Identity transitions as prior
        B = np.zeros((self.num_states, self.num_states, self.num_actions))
        for a in range(self.num_actions):
            B[:, :, a] = np.eye(self.num_states)
            
        self.agent.set_matrices(A=A, B=B)

    def set_preferences(self, preferred_observations):
        """
        Sets the preference vector C. preferred_observations is a dict of {obs_index: preference_weight}
        """
        C = np.ones(self.num_obs) * 0.01
        for obs_idx, weight in preferred_observations.items():
            if 0 <= obs_idx < self.num_obs:
                C[obs_idx] = weight
        self.agent.set_matrices(C=C)

    def get_llm_response(self, prompt):
        """
        Simulates or executes an LLM call to process cognitive states.
        If a client is provided, runs actual completion. Otherwise, uses mathematical fallback.
        """
        if self.llm_client:
            try:
                # Assuming standard client schema (e.g. OpenAI or similar)
                response = self.llm_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            except Exception as e:
                self.logger.warning(f"LLM API Call failed: {e}. Falling back to simulated reasoning.")
        
        # Simulated fallback based on current state probabilities
        current_state = np.argmax(self.agent.q)
        state_name = self.states_desc[current_state]
        return json.dumps({
            "thought": f"The agent feels it is currently in state: {state_name}. Minimizing free energy.",
            "predicted_observation": self.observations_desc[current_state],
            "confidence": float(self.agent.q[current_state])
        })

    def run_step(self, actual_obs_index):
        """
        Runs a single active inference step:
        1. Infer state from actual observation
        2. Calculate surprise (Variational Free Energy)
        3. Query LLM to reflect on current state and surprise
        4. Select next action to minimize Expected Free Energy
        """
        # 1. Update beliefs based on observation
        q = self.agent.infer_states(actual_obs_index)
        vfe = self.agent.calculate_vfe(actual_obs_index)
        
        # 2. Check for surprise (VFE spike)
        is_surprised = vfe > 1.5
        surprise_msg = "HIGH SURPRISE (World model mismatch)" if is_surprised else "Low surprise (Expected transition)"
        
        self.logger.info(f"Observed: {self.observations_desc[actual_obs_index]} | VFE: {vfe:.4f} ({surprise_msg})")
        
        # 3. Construct prompt for LLM cognitive reflection
        prompt = (
            f"You are the cognitive mind of an Active Inference agent.\n"
            f"Possible States: {self.states_desc}\n"
            f"Possible Actions: {self.actions_desc}\n"
            f"Current Belief Distribution: {q.tolist()}\n"
            f"Current Observation: {self.observations_desc[actual_obs_index]}\n"
            f"Variational Free Energy (Surprise): {vfe:.4f}\n"
            f"Reflect on this observation and formulate a cognitive thought."
        )
        
        llm_reflection = self.get_llm_response(prompt)
        self.logger.info(f"LLM Reflection: {llm_reflection}")
        
        # 4. Action Selection
        action, action_probs = self.agent.select_action()
        self.logger.info(f"Selected action: {self.actions_desc[action]} | Action distribution: {action_probs.tolist()}")
        
        return {
            "belief": q,
            "vfe": vfe,
            "reflection": llm_reflection,
            "selected_action": action,
            "action_desc": self.actions_desc[action]
        }
