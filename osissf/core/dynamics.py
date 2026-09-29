import numpy as np
import pinocchio as pin
from .ur5e_dynamics_data import MASSES, COMS, INERTIA_TENSORS

class PinocchioDynamicsCalculator:
    def __init__(self, urdf_path, ee_link_name):
        print("🔧 Menginisialisasi Pinocchio...")
        self.model = pin.buildModelFromUrdf(urdf_path)

        # Inject Data Dinamika YAML
        for i in range(self.model.nv):
            # joint_id 1 corresponds to the first moving link
            self.model.inertias[i + 1] = pin.Inertia(
                MASSES[i], COMS[i], INERTIA_TENSORS[i]
            )

        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(ee_link_name)
        print(
            f"✅ Pinocchio Siap. Target Frame: {ee_link_name} (ID: {self.ee_frame_id})"
        )

    def get_jacobian(self, q):
        """
        Menghitung Jacobian dan membaliknya agar sesuai dengan frame RTDE
        RTDE X+ adalah Belakang, Pinocchio X+ adalah Depan.
        """
        pin.forwardKinematics(self.model, self.data, q)
        J_world = pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        J_base = J_world[:3, :].copy()
        J_base[0, :] = -J_base[0, :]  # Flip X
        J_base[1, :] = -J_base[1, :]  # Flip Y
        return J_base

    def get_mass_matrix(self, q):
        pin.crba(self.model, self.data, q)
        return self.data.M.copy()

    def get_nullspace(self, q, J):
        M = self.get_mass_matrix(q)
        M_inv = np.linalg.inv(M)
        try:
            Lambda = np.linalg.inv(J @ M_inv @ J.T)
        except np.linalg.LinAlgError:
            A = J @ M_inv @ J.T
            Lambda = np.linalg.pinv(A)
        # J_bar = M^-1 * J.T * (J * M^-1 * J.T)^-1
        J_bar = M_inv @ J.T @ Lambda
        eye = np.eye(self.model.nv)  # Identity Matrix
        NT = eye - (J.T @ J_bar.T)
        return NT, M_inv

    def get_coriolis_gravity(self, q, dq):
        """
        Menghitung vektor Coriolis dan gravitasi
        """
        # Set acceleration to zero for gravity/Coriolis compensation
        ddq = np.zeros_like(dq)
        # Compute forward kinematics first
        pin.forwardKinematics(self.model, self.data, q, dq, ddq)
        pin.updateFramePlacements(self.model, self.data)
        # RNEA: Recursive Newton-Euler Algorithm
        # Arguments: model, data, q (position), dq (velocity), ddq (acceleration)
        pin.rnea(self.model, self.data, q, dq, ddq)
        # tau = C(q,dq)*dq + g(q), where C is Coriolis matrix and g is gravity vector
        # The rnea function already computes both terms together
        return self.data.tau.copy()