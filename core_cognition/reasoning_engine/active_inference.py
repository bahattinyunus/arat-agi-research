import numpy as np

class ActiveInferenceAgent:
    """
    Active Inference Agent based on Karl Friston's Free Energy Principle.
    Models decision making under uncertainty using POMDP formulation.
    """
    def __init__(self, num_states, num_obs, num_actions):
        self.num_states = num_states
        self.num_obs = num_obs
        self.num_actions = num_actions

        # A Matrix: Observation Likelihood P(o | s)
        # Dimensions: (num_obs, num_states)
        self.A = np.ones((num_obs, num_states)) / num_obs
        
        # B Matrix: Transition Probabilities P(s_t+1 | s_t, action)
        # Dimensions: (num_states, num_states, num_actions)
        self.B = np.ones((num_states, num_states, num_actions)) / num_states
        
        # C Matrix: Prior Preferences over observations (Target distribution)
        # Dimensions: (num_obs,)
        self.C = np.ones(num_obs) / num_obs
        
        # D Matrix: Prior Beliefs about initial states P(s_0)
        # Dimensions: (num_states,)
        self.D = np.ones(num_states) / num_states
        
        # Current belief state q(s)
        self.q = np.copy(self.D)

    def set_matrices(self, A=None, B=None, C=None, D=None):
        """
        Updates the generative model matrices. Normalizes to maintain probability constraints.
        """
        if A is not None:
            # Normalize column-wise so that sum_o P(o | s) = 1
            self.A = A / (np.sum(A, axis=0) + 1e-12)
        if B is not None:
            # Normalize column-wise for each action so that sum_s' P(s' | s, a) = 1
            for a in range(self.num_actions):
                self.B[:, :, a] = B[:, :, a] / (np.sum(B[:, :, a], axis=0) + 1e-12)
        if C is not None:
            # Normalize preferences to sum to 1
            self.C = C / (np.sum(C) + 1e-12)
        if D is not None:
            # Normalize prior state distribution
            self.D = D / (np.sum(D) + 1e-12)
            self.q = np.copy(self.D)

    def softmax(self, x):
        """Stable softmax function"""
        e_x = np.exp(x - np.max(x))
        return e_x / (np.sum(e_x) + 1e-12)

    def infer_states(self, observation):
        """
        Updates internal beliefs q(s) based on the observation received.
        Minimizes Variational Free Energy via fixed-point iteration.
        """
        # Log likelihood of the observation given states: ln P(o | s)
        log_likelihood = np.log(self.A[observation, :] + 1e-12)
        
        # Log prior: ln P(s)
        log_prior = np.log(self.q + 1e-12)
        
        # Fixed point iteration to minimize Variational Free Energy
        # q(s) = softmax( ln P(o | s) + ln P(s) )
        self.q = self.softmax(log_likelihood + log_prior)
        return self.q

    def calculate_vfe(self, observation):
        """
        Calculates Variational Free Energy (F) for the current state:
        F = E_q[ln q(s) - ln P(o, s)]
        F = KL(q(s) || P(s)) - E_q[ln P(o | s)]
        """
        # KL Divergence between posterior q(s) and prior D(s) (or previous belief)
        kl = np.sum(self.q * np.log((self.q + 1e-12) / (self.D + 1e-12)))
        
        # Expected log likelihood
        expected_log_lik = np.sum(self.q * np.log(self.A[observation, :] + 1e-12))
        
        vfe = kl - expected_log_lik
        return vfe

    def calculate_efe(self, action):
        """
        Calculates Expected Free Energy (G) for a specific action:
        G(a) = KL(q(o_t+1 | a) || P(o_t+1)) + E_q(s_t+1 | a)[ H(P(o_t+1 | s_t+1)) ]
        Where the first term is divergence from preferred outcomes (pragmatic utility),
        and the second is expected entropy of observations (epistemic value/uncertainty).
        """
        # Predict future state: P(s_t+1 | a) = B * q(s_t)
        q_next = np.dot(self.B[:, :, action], self.q)
        
        # Predict future observation: P(o_t+1 | a) = A * q_next
        o_next = np.dot(self.A, q_next)
        
        # 1. Divergence from preferences: KL( q(o_t+1 | a) || C )
        # Divergence = sum( q(o_t+1 | a) * (ln q(o_t+1 | a) - ln C) )
        kl_divergence = np.sum(o_next * np.log((o_next + 1e-12) / (self.C + 1e-12)))
        
        # 2. Epistemic value / Expected Entropy: E_q(s_t+1)[ H(P(o_t+1 | s_t+1)) ]
        # Entropy H = - sum_o( P(o | s) * ln P(o | s) )
        H_A = -np.sum(self.A * np.log(self.A + 1e-12), axis=0)
        expected_entropy = np.sum(q_next * H_A)
        
        # Expected Free Energy: G(a) = Pragmatic Utility + Epistemic Value
        # Note: Pragmatic Utility is minimized when divergence from preference is low.
        # Epistemic Value is minimized when expected entropy is low (resolving uncertainty).
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
            
        # Action probability is proportional to -EFE
        action_probs = self.softmax(-efe_values)
        
        # Select action stochastically based on probabilities
        selected_action = np.random.choice(self.num_actions, p=action_probs)
        return selected_action, action_probs
