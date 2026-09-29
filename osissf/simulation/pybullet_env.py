import numpy as np
import pybullet as p
import pybullet_data

from oscbf.core.ur5e_collision_model import ur5e_collision_data


def _draw_box(xyz_min, xyz_max, rgba):
    """
    Visualisasi box containment. Memakai fungsi yang sama dengan
    hil14 (oscbf.utils.pybullet_visuals.create_3d_box_visualization);
    fallback: box visual transparan + 12 rusuk.
    """
    xyz_min = [float(v) for v in xyz_min]
    xyz_max = [float(v) for v in xyz_max]
    try:
        from oscbf.utils.pybullet_visuals import (
            create_3d_box_visualization,
        )
        return create_3d_box_visualization(p, xyz_min, xyz_max, rgba)
    except ImportError:
        pass

    lo = np.asarray(xyz_min)
    hi = np.asarray(xyz_max)
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    if rgba[3] > 0.0:
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half.tolist(),
            rgbaColor=list(rgba),
        )
        p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=vis,
            basePosition=center.tolist(),
        )
    corners = [
        np.array([x, y, z])
        for x in (lo[0], hi[0])
        for y in (lo[1], hi[1])
        for z in (lo[2], hi[2])
    ]
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(np.abs(corners[i] - corners[j]) > 1e-9) == 1:
                p.addUserDebugLine(
                    corners[i].tolist(),
                    corners[j].tolist(),
                    lineColorRGB=list(rgba[:3]),
                    lineWidth=2,
                )


class PyBulletEnvironment:
    """
    Environment PyBullet untuk Digital Twin UR5e.

    Tanggung jawab:
    - membuka koneksi PyBullet
    - membuat plane dan meja
    - load UR5e
    - membuat obstacle
    - membuat target sphere
    - membuat actual EE sphere
    - membuat visual collision spheres robot
    - mengupdate object-object tersebut

    Tidak menangani:
    - controller
    - CBF
    - RTDE
    - trajectory generation
    """

    def __init__(
        self,
        urdf_path,
        obstacle_type="sphere",
        show_obstacle="orange",
        show_actual=False,
        show_env=True,
        gui=True,
        table_height=0.59,
        traj=None,
        cbf="both",
    ):
        self.urdf_path = urdf_path
        self.traj = traj
        self.cbf = cbf
        self.tube_radius = 0.055
        self.tube_height = 1.1
        self.obstacle_type = obstacle_type
        # CLI memakai "blue", environment memakai "purple"
        self.show_obstacle = (
            "purple" if show_obstacle == "blue" else show_obstacle
        )
        self.show_actual = show_actual
        self.show_env = show_env
        self.gui = gui
        self.table_height = table_height

        self.client_id = None

        self.robot_id = None
        self.sim_joint_indices = []

        self.target_sphere_id = None
        self.ee_actual_sphere_id = None

        self.robot_sphere_ids = []

        self.robot_base_position = np.array(
            [0.0, 0.0, table_height],
            dtype=float,
        )

        self.table_position = np.array(
            [0.35, 0.0, table_height / 2.0],
            dtype=float,
        )

        self.containment_params = None
        self.obstacle_params = None
        self.tube_params = []

    # =========================================================
    # SETUP
    # =========================================================

    def connect(self):
        if self.client_id is not None:
            return

        if self.gui:
            self.client_id = p.connect(
                p.GUI,
                options="--width=1920 --height=1080",
            )
        else:
            self.client_id = p.connect(
                p.DIRECT
            )

        if self.client_id < 0:
            raise RuntimeError(
                "Gagal membuka koneksi PyBullet."
            )
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )
        p.resetSimulation()
        p.setGravity(
            0.0,
            0.0,
            -9.81,
        )
        p.setRealTimeSimulation(0)
        p.setTimeStep(0.008)

    def setup(self):
        """
        Membuat seluruh environment.
        """
        self.connect()
        self._create_plane()
        self._create_table()
        self._load_robot()
        self._create_obstacles()
        self._create_target_sphere()
        self._create_actual_ee_sphere()
        self._create_robot_collision_spheres()
        self._configure_camera()

        return self

    # =========================================================
    # BASIC ENVIRONMENT
    # =========================================================

    def _create_plane(self):
        p.loadURDF(
            "plane.urdf"
        )

    def _create_table(self):
        table_half_extents = [
            1.7 / 2.0,
            1.0 / 2.0,
            self.table_height / 2.0,
        ]

        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=table_half_extents,
            rgbaColor=[ 0.55, 0.27, 0.07, 1.0],
        )

        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=table_half_extents,
        )

        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=self.table_position.tolist(),
        )

    def _load_robot(self):
        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=self.robot_base_position.tolist(),
            useFixedBase=True,
        )

        self.sim_joint_indices = []

        for i in range(
            p.getNumJoints(self.robot_id)
        ):
            joint_info = p.getJointInfo(
                self.robot_id,
                i,
            )

            if joint_info[2] != p.JOINT_FIXED:
                self.sim_joint_indices.append(i)

    # =========================================================
    # OBSTACLE ENVIRONMENT
    # =========================================================

    def _create_obstacles(self):
        self.obstacle_params = {
            "size": [0.3, 0.8, self.table_height * 3.0],
            "position": [ 0.8, 0.4, self.table_height * 1.75],
            "color": [1.0, 0.6, 0.6, 0.25],
        }

        tube_height = self.tube_height
        tube_radius = self.tube_radius
        tube_z_center = (
            self.table_height + tube_height / 2.0
        )

        num_spheres = (
            int(tube_height / (tube_radius * 2.0)) + 1
        )

        self.tube_params = []
        if self.obstacle_type == "tube":
            if self.show_obstacle in ["purple", "both"]:
                purple = {
                    "position": [0.703, 0.350, tube_z_center],
                    "color": [0.6, 0.3, 0.8, 0.6],
                    "sphere_positions": [],
                }

                for i in range(num_spheres):
                    z_pos = (self.table_height + (i * tube_height / max(1, num_spheres - 1)))
                    purple["sphere_positions"].append([0.703, 0.350, z_pos])
                self.tube_params.append(purple)

            if self.show_obstacle in ["orange", "both",]:
                orange = {
                    "position": [0.596, -0.260, tube_z_center],
                    "color": [0.8, 0.4, 0.2, 0.6],
                    "sphere_positions": [],
                }

                for i in range(
                    num_spheres
                ):
                    z_pos = (
                        self.table_height + (i * tube_height / max(1, num_spheres - 1))
                    )
                    orange["sphere_positions"].append([0.596, -0.260, z_pos])
                self.tube_params.append(
                    orange
                )

        self.containment_params = {
            "xyz_min": [
                -0.3,
                -0.60,
                self.table_height + 0.2,
            ],
            "xyz_max": [
                0.70,
                0.45,
                self.table_height + 0.8,
            ],
            "wb_xyz_min": [
                -0.37,
                -0.35,
                self.table_height - 0.2,
            ],
            "wb_xyz_max": [
                0.71,
                0.46,
                self.table_height + 0.85,
            ],
            "color": (0.3, 0.9, 0.3, 0.4),
            "color_wb": (0.2, 0.5, 0.9, 0.20),
            "tube_params": self.tube_params,
            "tube_radius": tube_radius,
            "tube_height": tube_height,
        }

        # hil14: override khusus circle_3d
        if self.traj == "circle_3d":
            self.containment_params["xyz_min"] = [
                -0.3, -0.60, self.table_height + 0.3,
            ]
            self.containment_params["wb_xyz_max"] = [
                0.75, 0.46, self.table_height + 0.85,
            ]

        if self.show_env:
            self._visualize_obstacles()

    def _visualize_obstacles(self):
        if not self.show_env:
            return

        cp = self.containment_params

        # 1. EE containment box (hijau) -- selalu, seperti hil14
        _draw_box(cp["xyz_min"], cp["xyz_max"], cp["color"])
        print(
            f"🟩 EE containment box (PyBullet): "
            f"min={cp['xyz_min']}, max={cp['xyz_max']}"
        )

        # hil14: whole-body box & obstacle hanya digambar jika --cbf both
        if self.cbf != "both":
            return

        # 2. Whole-body containment box (biru)
        _draw_box(cp["wb_xyz_min"], cp["wb_xyz_max"], cp["color_wb"])
        print(
            f"🟦 Whole-body box (PyBullet): "
            f"min={cp['wb_xyz_min']}, max={cp['wb_xyz_max']}"
        )

        # -----------------------------------------------------
        # Outer sphere obstacle
        # -----------------------------------------------------

        if self.obstacle_type == "sphere":
            radius = min(
                self.obstacle_params["size"][0],
                self.obstacle_params["size"][1],
            ) / 2.0

            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=radius,
                rgbaColor=self.obstacle_params[
                    "color"
                ],
            )

            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=self.obstacle_params[
                    "position"
                ],
            )

            # ISSf visual margin
            issf_margin = 0.03
            inner_radius = (
                radius - issf_margin
            )

            if inner_radius > 0.02:
                inner_shape_id = (
                    p.createVisualShape(
                        shapeType=p.GEOM_SPHERE,
                        radius=inner_radius,
                        rgbaColor=[0.5, 0.0, 0.0, 0.55,],
                    )
                )

                p.createMultiBody(
                    baseMass=0.0,
                    baseVisualShapeIndex=inner_shape_id,
                    basePosition=self.obstacle_params[
                        "position"
                    ],
                )

        # -----------------------------------------------------
        # Tube obstacle
        # -----------------------------------------------------

        elif self.obstacle_type == "tube":
            for tube in self.tube_params:
                visual_shape_id = (
                    p.createVisualShape(
                        shapeType=p.GEOM_SPHERE,
                        radius=self.containment_params[
                            "tube_radius"
                        ],
                        rgbaColor=tube["color"],
                    )
                )

                for position in tube[
                    "sphere_positions"
                ]:
                    p.createMultiBody(
                        baseMass=0.0,
                        baseVisualShapeIndex=visual_shape_id,
                        basePosition=position,
                    )

    # =========================================================
    # TARGET SPHERE
    # =========================================================

    def _create_target_sphere(self):
        if self.obstacle_type is None:
            return

        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.03,
            rgbaColor=[ 1.0, 1.0, 0.0, 0.8,],
        )

        collision_shape_id = (
            p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=0.03,
            )
        )

        self.target_sphere_id = (
            p.createMultiBody(
                baseMass=0.01,
                baseVisualShapeIndex=visual_shape_id,
                baseCollisionShapeIndex=collision_shape_id,
                basePosition=[ 0.0, 0.0, 0.0,],
            )
        )

        p.changeDynamics(
            self.target_sphere_id,
            -1,
            linearDamping=10.0,
            angularDamping=10.0,
        )

        # Disable collision target-vs-robot
        for joint_idx in range(
            -1,
            p.getNumJoints(
                self.robot_id
            ),
        ):
            p.setCollisionFilterPair(
                self.robot_id,
                self.target_sphere_id,
                joint_idx,
                -1,
                0,
            )

    # =========================================================
    # ACTUAL EE SPHERE
    # =========================================================

    def _create_actual_ee_sphere(self):

        if not self.show_actual:
            self.ee_actual_sphere_id = None
            return

        visual_shape_id = (
            p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.015,
                rgbaColor=[ 0.0, 0.0, 1.0, 0.8,],
            )
        )

        self.ee_actual_sphere_id = (
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=[0.0, 0.0, 0.0,],
            )
        )

    # =========================================================
    # ROBOT COLLISION SPHERES
    # =========================================================

    def _create_robot_collision_spheres(self):

        all_radii = np.concatenate(
            ur5e_collision_data["radii"]
        )
        self.robot_sphere_ids = []
        for radius in all_radii:
            visual_shape_id = (
                p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=float(radius),
                    rgbaColor=[0.2, 0.8, 0.2, 0.9],
                )
            )

            body_id = p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=[0.0, 0.0, -1.0],
            )

            self.robot_sphere_ids.append(
                body_id
            )

    # =========================================================
    # CAMERA
    # =========================================================

    def _configure_camera(self):
        if not self.gui:
            return

        p.resetDebugVisualizerCamera(
            cameraDistance=1.8,
            cameraYaw=135,
            cameraPitch=-20,
            cameraTargetPosition=[
                0.0,
                0.0,
                self.robot_base_position[2]
                + 0.2,
            ],
        )

        p.configureDebugVisualizer(
            p.COV_ENABLE_GUI,
            0,
        )

    # =========================================================
    # OBJECT UPDATE
    # =========================================================

    def update_target(
        self,
        x_des,
    ):
        """
        x_des berasal dari RTDE frame.
        Convert:
            RTDE -> PyBullet
        """
        if (
            self.target_sphere_id is None
            or not p.isConnected()
        ):
            return

        x_des = np.asarray(
            x_des,
            dtype=float,
        ).copy()

        x_des[0] *= -1.0
        x_des[1] *= -1.0
        world_position = (x_des + self.robot_base_position)
        p.resetBasePositionAndOrientation(
            self.target_sphere_id,
            world_position.tolist(),
            [0, 0, 0, 1],
        )

    def update_actual_ee(
        self,
        x_actual,
    ):
        if (
            self.ee_actual_sphere_id is None
            or not p.isConnected()
        ):
            return

        x_actual = np.asarray(
            x_actual,
            dtype=float,
        ).copy()

        x_actual[0] *= -1.0
        x_actual[1] *= -1.0
        world_position = (
            x_actual
            + self.robot_base_position
        )

        p.resetBasePositionAndOrientation(
            self.ee_actual_sphere_id,
            world_position.tolist(),
            [0, 0, 0, 1],
        )

    def update_robot_collision_spheres(
        self,
        robot_collision_data,
    ):
        """
        robot_collision_data:
            array (N, 4):
            [x, y, z, radius]
        Posisi berasal dari Manipulator.link_collision_data().
        """

        if not p.isConnected():
            return

        count = min(
            len(self.robot_sphere_ids),
            robot_collision_data.shape[0],
        )

        for i in range(count):

            position_pino = (
                robot_collision_data[i, :3]
            )

            world_position = (
                position_pino
                + self.robot_base_position
            )

            p.resetBasePositionAndOrientation(
                self.robot_sphere_ids[i],
                world_position.tolist(),
                [0, 0, 0, 1],
            )

    # =========================================================
    # JOINT CONTROL
    # =========================================================

    def apply_joint_velocity(
        self,
        dq_cmd,
        max_force=150.0,
    ):
        """
        Simulasikan speedJ menggunakan
        PyBullet velocity control.
        """
        dq_cmd = np.asarray(
            dq_cmd,
            dtype=float,
        )

        for i, joint_idx in enumerate(
            self.sim_joint_indices
        ):

            p.setJointMotorControl2(
                bodyIndex=int(
                    self.robot_id
                ),
                jointIndex=int(
                    joint_idx
                ),
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=float(
                    dq_cmd[i]
                ),
                force=max_force,
            )

    def step(self):
        if p.isConnected():
            p.stepSimulation()

    # =========================================================
    # STATE
    # =========================================================

    def get_joint_state(self):
        """
        Mengambil posisi dan kecepatan joint UR5e dari PyBullet.

        Returns
        -------
        q : np.ndarray, shape (6,)
            Joint position.
        dq : np.ndarray, shape (6,)
            Joint velocity.
        """
        if not p.isConnected():
            raise RuntimeError(
                "PyBullet belum terhubung."
            )

        states = p.getJointStates(
            self.robot_id,
            self.sim_joint_indices,
        )

        q = np.asarray(
            [state[0] for state in states],
            dtype=float,
        )

        dq = np.asarray(
            [state[1] for state in states],
            dtype=float,
        )

        return q, dq

    # =========================================================
    # STATE
    # =========================================================

    def reset_joint_state(
        self,
        q,
        dq=None,
    ):
        """
        Set state robot PyBullet secara langsung.
        Digunakan untuk initialization / homing simulation,
        bukan sebagai control command.
        """

        if not p.isConnected():
            raise RuntimeError(
                "PyBullet belum terhubung."
            )

        q = np.asarray(
            q,
            dtype=float,
        )

        if dq is None:
            dq = np.zeros_like(q)

        dq = np.asarray(
            dq,
            dtype=float,
        )

        for i, joint_idx in enumerate(
            self.sim_joint_indices
        ):
            p.resetJointState(
                bodyUniqueId=self.robot_id,
                jointIndex=int(joint_idx),
                targetValue=float(q[i]),
                targetVelocity=float(dq[i]),
            )

        def get_joint_state(self):
            if not p.isConnected():
                raise RuntimeError(
                    "PyBullet belum terhubung."
                )
            states = p.getJointStates(
                self.robot_id,
                self.sim_joint_indices,
            )
            q = np.asarray(
                [
                    state[0]
                    for state in states
                ],
                dtype=float,
            )
            dq = np.asarray(
                [
                    state[1]
                    for state in states
                ],
                dtype=float,
            )
            return q, dq

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):
        if p.isConnected():
            p.disconnect()

        self.client_id = None