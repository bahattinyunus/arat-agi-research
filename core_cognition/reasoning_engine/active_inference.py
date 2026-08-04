import numpy as np

class ActiveInferenceAgent:
    """
    Active Inference Agent based on Karl Friston's Free Energy Principle.
    Models decision making under uncertainty using POMDP formulation.
    Supports online Dirichlet learning of likelihood (A) and transition (B) matrices.
    """
    def __init__(self, num_states, num_obs, num_actions, learning_rate=0.1):
        self.num_states = num_states
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.lr = learning_rate

        # Dirichlet accumulators for learning: initialized to 1.0 (flat prior)
        self.a_counts = np.ones((num_obs, num_states))
        self.b_counts = np.ones((num_states, num_states, num_actions))

        # A Matrix: Observation Likelihood P(o | s)
        self.A = np.copy(self.a_counts) / num_obs
        
        # B Matrix: Transition Probabilities P(s_t+1 | s_t, action)
        self.B = np.copy(self.b_counts) / num_states
        
        # C Matrix: Prior Preferences over observations (Target distribution)
        self.C = np.ones(num_obs) / num_obs
        
        # D Matrix: Prior Beliefs about initial states P(s_0)
        self.D = np.ones(num_states) / num_states
        
        # Current belief state q(s)
        self.q = np.copy(self.D)
        self.prev_q = np.copy(self.D)
        self.prev_action = None

    def set_matrices(self, A=None, B=None, C=None, D=None):
        """
        Updates the generative model matrices. Normalizes to maintain probability constraints.
        Also resets Dirichlet accumulation variables to match the new matrices.
        """
        if A is not None:
            self.A = A / (np.sum(A, axis=0) + 1e-12)
            self.a_counts = np.copy(self.A) * 10.0 # scale counts to give weight to setting
        if B is not None:
            for a in range(self.num_actions):
                self.B[:, :, a] = B[:, :, a] / (np.sum(B[:, :, a], axis=0) + 1e-12)
            self.b_counts = np.copy(self.B) * 10.0
        if C is not None:
            self.C = C / (np.sum(C) + 1e-12)
        if D is not None:
            self.D = D / (np.sum(D) + 1e-12)
            self.q = np.copy(self.D)
            self.prev_q = np.copy(self.D)

    def softmax(self, x):
        """Stable softmax function"""
        e_x = np.exp(x - np.max(x))
        return e_x / (np.sum(e_x) + 1e-12)

    def infer_states(self, observation):
        """
        Updates internal beliefs q(s) based on the observation received.
        Minimizes Variational Free Energy via fixed-point iteration.
        """
        self.prev_q = np.copy(self.q)
        
        # Log likelihood of the observation given states: ln P(o | s)
        log_likelihood = np.log(self.A[observation, :] + 1e-12)
        
        # Log prior: ln P(s)
        log_prior = np.log(self.q + 1e-12)
        
        # Fixed point iteration to minimize Variational Free Energy
        # q(s) = softmax( ln P(o | s) + ln P(s) )
        self.q = self.softmax(log_likelihood + log_prior)
        return self.q

    def update_model(self, actual_obs):
        """
        Online model learning.
        Updates Dirichlet count accumulators for Likelihood (A) and Transition (B) matrices
        based on inferred states and observations.
        """
        # 1. Update Likelihood matrix counts A[o, s] using current belief q(s)
        # We observed actual_obs under belief state q
        o_vec = np.zeros(self.num_obs)
        o_vec[actual_obs] = 1.0
        
        # Update Dirichlet parameters: a_counts += lr * (o_one_hot x q^T)
        self.a_counts += self.lr * np.outer(o_vec, self.q)
        # Re-normalize columns of A
        self.A = self.a_counts / (np.sum(self.a_counts, axis=0) + 1e-12)

        # 2. Update Transition matrix counts B[s_t+1, s_t, action_t] using transition belief
        if self.prev_action is not None:
            # Update B counts: b_counts[:, :, action] += lr * (q_t x q_t-1^T)
            self.b_counts[:, :, self.prev_action] += self.lr * np.outer(self.q, self.prev_q)
            # Re-normalize transition columns
            self.B[:, :, self.prev_action] = self.b_counts[:, :, self.prev_action] / (
                np.sum(self.b_counts[:, :, self.prev_action], axis=0) + 1e-12
            )

    def calculate_vfe(self, observation):
        """
        Calculates Variational Free Energy (F) for the current state:
        F = E_q[ln q(s) - ln P(o, s)]
        F = KL(q(s) || P(s)) - E_q[ln P(o | s)]
        """
        kl = np.sum(self.q * np.log((self.q + 1e-12) / (self.D + 1e-12)))
        expected_log_lik = np.sum(self.q * np.log(self.A[observation, :] + 1e-12))
        vfe = kl - expected_log_lik
        return vfe

    def calculate_efe(self, action):
        """
        Calculates Expected Free Energy (G) for a specific action:
        G(a) = KL(q(o_t+1 | a) || P(o_t+1)) + E_q(s_t+1 | a)[ H(P(o_t+1 | s_t+1)) ]
        """
        q_next = np.dot(self.B[:, :, action], self.q)
        o_next = np.dot(self.A, q_next)
        
        # 1. Divergence from preferences: KL( q(o_t+1 | a) || C )
        kl_divergence = np.sum(o_next * np.log((o_next + 1e-12) / (self.C + 1e-12)))
        
        # 2. Epistemic value / Expected Entropy: E_q(s_t+1)[ H(P(o_t+1 | s_t+1)) ]
        H_A = -np.sum(self.A * np.log(self.A + 1e-12), axis=0)
        expected_entropy = np.sum(q_next * H_A)
        
        efe = kl_divergence + expected_entropy
        return efe

    def select_action(self):
        """
        Selects the action that minimizes Expected Free Energy (EFE).
        Uses a softmax policy over negative EFE.
        """
        efe_values = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            efe_values[a] = self.calculate_efe(a)
            
        action_probs = self.softmax(-efe_values)
        selected_action = np.random.choice(self.num_actions, p=action_probs)
        
        # Store for learning in next step
        self.prev_action = selected_action
        return selected_action, action_probs


class HierarchicalActiveInferenceAgent:
    """
    Hierarchical Active Inference Agent with two levels based on Friston's formulation:
    - High-level (Goal/Context layer)
    - Low-level (Reactive/Action layer)
    The high-level agent infers abstract states and selects abstract goal actions.
    These goal actions set the prior preference (C) vector for the low-level agent.
    """
    def __init__(self, high_params, low_params, goal_to_preference_matrix):
        """
        high_params: dict with keys (num_states, num_obs, num_actions) and optionally learning_rate
        low_params: dict with keys (num_states, num_obs, num_actions) and optionally learning_rate
        goal_to_preference_matrix: numpy array of shape (high_num_actions, low_num_obs)
            Maps high-level selected goal actions to preferred observations for the low level.
        """
        self.high_agent = ActiveInferenceAgent(
            high_params["num_states"], 
            high_params["num_obs"], 
            high_params["num_actions"],
            learning_rate=high_params.get("learning_rate", 0.1)
        )
        self.low_agent = ActiveInferenceAgent(
            low_params["num_states"], 
            low_params["num_obs"], 
            low_params["num_actions"],
            learning_rate=low_params.get("learning_rate", 0.1)
        )
        self.goal_to_pref = np.array(goal_to_preference_matrix)

    def run_step(self, high_obs, low_obs):
        """
        Runs one step of hierarchical inference:
        1. High-level agent infers high-level state from high_obs
        2. High-level agent selects high-level goal action
        3. Low-level agent's C matrix (preferences) is set based on the selected goal action
        4. Low-level agent infers state from low_obs
        5. Low-level agent selects low-level action
        """
        # High level inference and action selection
        self.high_agent.infer_states(high_obs)
        high_action, high_action_probs = self.high_agent.select_action()
        self.high_agent.update_model(high_obs)

        # Update low level preferences (C matrix)
        preferred_obs_dist = self.goal_to_pref[high_action]
        self.low_agent.set_matrices(C=preferred_obs_dist)

        # Low level inference and action selection
        self.low_agent.infer_states(low_obs)
        low_action, low_action_probs = self.low_agent.select_action()
        self.low_agent.update_model(low_obs)

        return {
            "high_action": high_action,
            "high_action_probs": high_action_probs,
            "low_action": low_action,
            "low_action_probs": low_action_probs,
            "high_belief": self.high_agent.q,
            "low_belief": self.low_agent.q
        }

