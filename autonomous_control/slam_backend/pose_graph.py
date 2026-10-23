import numpy as np
from typing import List, Dict, Tuple, Optional, Any

class PoseGraphEdge:
    def __init__(self, from_idx: int, to_idx: int, measurement: Tuple[float, float, float], information_diag: Tuple[float, float, float]):
        self.from_idx = from_idx
        self.to_idx = to_idx
        self.dx = float(measurement[0])
        self.dy = float(measurement[1])
        self.dtheta = float(measurement[2])
        self.info = np.diag(information_diag)

class PoseGraphOptimizer:
    """
    2D Pose Graph Optimization SLAM Backend using Gauss-Newton Non-Linear Least Squares.
    Solves loop closures and corrects accumulated odometry drift across exploratory trajectories:
    F(X) = sum_ij ( e_ij(X)^T * Omega_ij * e_ij(X) )
    (J^T * Omega * J) * delta_X = -J^T * Omega * e
    """
    def __init__(self):
        self.poses: List[np.ndarray] = []  # List of [x, y, theta] arrays
        self.edges: List[PoseGraphEdge] = []

    def add_node(self, pose: Tuple[float, float, float]) -> int:
        idx = len(self.poses)
        self.poses.append(np.array(pose, dtype=float))
        return idx

    def add_edge(
        self,
        from_idx: int,
        to_idx: int,
        measurement: Tuple[float, float, float],
        information_diag: Tuple[float, float, float] = (100.0, 100.0, 500.0)
    ):
        edge = PoseGraphEdge(from_idx, to_idx, measurement, information_diag)
        self.edges.append(edge)

    def compute_edge_residual_and_jacobian(self, edge: PoseGraphEdge) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes error vector e_ij and Jacobians J_i, J_j for relative pose measurement:
        e_ij = R_i^T * (p_j - p_i) - d_meas
        theta_err = wrap(theta_j - theta_i - dtheta_meas)
        """
        xi = self.poses[edge.from_idx]
        xj = self.poses[edge.to_idx]

        xi_x, xi_y, xi_th = xi[0], xi[1], xi[2]
        xj_x, xj_y, xj_th = xj[0], xj[1], xj[2]

        c = np.cos(xi_th)
        s = np.sin(xi_th)
        R_i = np.array([[c, -s], [s, c]])

        d_pos = np.array([xj_x - xi_x, xj_y - xi_y])
        pred_d = R_i.T @ d_pos

        pred_dth = xj_th - xi_th
        pred_dth = np.arctan2(np.sin(pred_dth), np.cos(pred_dth))

        err_pos = pred_d - np.array([edge.dx, edge.dy])
        err_th = np.arctan2(np.sin(pred_dth - edge.dtheta), np.cos(pred_dth - edge.dtheta))
        e = np.array([err_pos[0], err_pos[1], err_th])

        # Jacobian w.r.t xi
        J_i = np.array([
            [-c, -s,  -s * (xj_x - xi_x) + c * (xj_y - xi_y)],
            [ s, -c,  -c * (xj_x - xi_x) - s * (xj_y - xi_y)],
            [ 0,  0,  -1.0]
        ])

        # Jacobian w.r.t xj
        J_j = np.array([
            [ c,  s,  0.0],
            [-s,  c,  0.0],
            [ 0,  0,  1.0]
        ])

        return e, J_i, J_j

    def optimize(self, max_iterations: int = 15, tolerance: float = 1e-4) -> float:
        """
        Executes Gauss-Newton iterations while keeping initial anchor pose (pose 0) fixed.
        Returns final total residual error.
        """
        num_nodes = len(self.poses)
        if num_nodes < 2 or not self.edges:
            return 0.0

        total_dim = num_nodes * 3

        for iteration in range(max_iterations):
            H = np.zeros((total_dim, total_dim))
            b = np.zeros(total_dim)
            total_error = 0.0

            for edge in self.edges:
                e, J_i, J_j = self.compute_edge_residual_and_jacobian(edge)
                omega = edge.info
                total_error += float(e.T @ omega @ e)

                i = edge.from_idx
                j = edge.to_idx

                # Submatrices
                H_ii = J_i.T @ omega @ J_i
                H_ij = J_i.T @ omega @ J_j
                H_ji = J_j.T @ omega @ J_i
                H_jj = J_j.T @ omega @ J_j

                b_i = J_i.T @ omega @ e
                b_j = J_j.T @ omega @ e

                # Assemble into global linear system
                H[i*3:i*3+3, i*3:i*3+3] += H_ii
                H[i*3:i*3+3, j*3:j*3+3] += H_ij
                H[j*3:j*3+3, i*3:i*3+3] += H_ji
                H[j*3:j*3+3, j*3:j*3+3] += H_jj

                b[i*3:i*3+3] += b_i
                b[j*3:j*3+3] += b_j

            # Anchor constraint: Fix pose 0 exactly using Dirichlet boundary condition
            H[0:3, :] = 0.0
            H[:, 0:3] = 0.0
            H[0:3, 0:3] = np.eye(3)
            b[0:3] = 0.0

            # Solve normal equations: H * delta_x = -b
            try:
                delta_x = np.linalg.solve(H, -b)
            except np.linalg.LinAlgError:
                delta_x = np.linalg.lstsq(H, -b, rcond=None)[0]

            # Update poses
            max_delta = 0.0
            for n in range(num_nodes):
                self.poses[n] += delta_x[n*3:n*3+3]
                self.poses[n][2] = np.arctan2(np.sin(self.poses[n][2]), np.cos(self.poses[n][2]))
                max_delta = max(max_delta, np.max(np.abs(delta_x[n*3:n*3+3])))

            if max_delta < tolerance:
                break

        return total_error

    def get_poses(self) -> List[List[float]]:
        return [p.tolist() for p in self.poses]
