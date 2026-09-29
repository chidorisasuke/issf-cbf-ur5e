import numpy as np


RTDE_AXIS_SIGN = np.array(
    [-1.0, -1.0, 1.0],
    dtype=float,
)


def pybullet_to_rtde(
    position,
    robot_base_position,
):
    """
    Convert posisi dari frame/world PyBullet
    ke frame RTDE yang digunakan oleh CBF.

    Implementasi mengikuti hil14:
        p_rtde =
            (p_pyb - p_robot_base)
            * [-1, -1, 1]
    """
    position = np.asarray(
        position,
        dtype=float,
    )
    robot_base_position = np.asarray(
        robot_base_position,
        dtype=float,
    )
    return (
        position - robot_base_position
    ) * RTDE_AXIS_SIGN


class SphereObstacle:
    """
    Representasi obstacle berbentuk sphere
    untuk safety/collision layer.
    """
    def __init__(
        self,
        position,
        radius,
        name=None,
    ):
        self.position = np.asarray(
            position,
            dtype=float,
        )
        self.radius = float(radius)
        self.name = name

    def to_rtde(
        self,
        robot_base_position,
        radius_margin=0.0,
    ):
        """
        Convert obstacle ke format RTDE.
        Returns:
            position_rtde
            radius_with_margin
        """
        position_rtde = pybullet_to_rtde(
            self.position,
            robot_base_position,
        )
        radius_rtde = (
            self.radius
            + radius_margin
        )
        return (
            position_rtde,
            radius_rtde,
        )


class TubeObstacle:
    """
    Representasi tube sebagai kumpulan sphere.
    dimana tube didiskritisasi menjadi beberapa
    sphere sepanjang sumbu Z.
    """

    def __init__(
        self,
        sphere_positions,
        radius,
        name=None,
    ):
        self.sphere_positions = [
            np.asarray(
                position,
                dtype=float,
            )
            for position in sphere_positions
        ]
        self.radius = float(radius)
        self.name = name

    def to_rtde(
        self,
        robot_base_position,
        radius_margin=0.0,
    ):
        """
        Convert seluruh sphere tube ke frame RTDE.
        Returns:
            positions_rtde
            radii
        """
        positions_rtde = [
            pybullet_to_rtde(
                position,
                robot_base_position,
            )
            for position in self.sphere_positions
        ]

        radii = [
            self.radius + radius_margin
            for _ in self.sphere_positions
        ]
        return (
            positions_rtde,
            radii,
        )


class CollisionEnvironment:
    """
    Mengelola seluruh obstacle collision environment.
    Tidak bergantung pada PyBullet secara langsung.
    """

    def __init__(
        self,
        robot_base_position,
        radius_margin=0.02,
    ):
        self.robot_base_position = np.asarray(
            robot_base_position,
            dtype=float,
        )

        self.radius_margin = float(
            radius_margin
        )
        self.obstacles = []

    def add_sphere(
        self,
        position,
        radius,
        name=None,
    ):
        obstacle = SphereObstacle(
            position=position,
            radius=radius,
            name=name,
        )
        self.obstacles.append(
            obstacle
        )

    def add_tube(
        self,
        sphere_positions,
        radius,
        name=None,
    ):
        obstacle = TubeObstacle(
            sphere_positions=sphere_positions,
            radius=radius,
            name=name,
        )
        self.obstacles.append(
            obstacle
        )

    def get_obstacle_arrays(self):
        """
        Menghasilkan obstacle position dan radius
        dalam format yang dibutuhkan CBF.
        Returns:
            collision_positions: (N, 3)
            collision_radii: (N,)
        """
        positions = []
        radii = []
        for obstacle in self.obstacles:
            obstacle_positions, obstacle_radii = (
                obstacle.to_rtde(
                    self.robot_base_position,
                    self.radius_margin,
                )
            )
            positions.extend(
                obstacle_positions
            )
            radii.extend(
                obstacle_radii
            )

        # Fallback agar dimensi JAX tetap valid
        # jika tidak ada obstacle.
        if len(positions) == 0:
            positions.append(
                np.array(
                    [100.0, 100.0, 100.0]
                )
            )
            radii.append(0.1)

        return (
            np.asarray(
                positions,
                dtype=float,
            ),
            np.asarray(
                radii,
                dtype=float,
            ),
        )

# ============================================================
# LINK 1: SHOULDER
# ============================================================

shoulder_spheres_pos = (
    (0.0, 0.0, 0.07),
)

shoulder_spheres_rad = (
    0.07,
)


# ============================================================
# LINK 2: UPPER ARM
# ============================================================

upper_arm_spheres_pos = (
    (-0.10, 0.0, 0.138),
    (-0.25, 0.0, 0.138),
)

upper_arm_spheres_rad = (
    0.06,
    0.06,
)


# ============================================================
# LINK 3: FOREARM
# ============================================================

forearm_spheres_pos = (
    (-0.15, 0.0, 0.007),
    (-0.25, 0.0, 0.007),
    (-0.35, 0.0, 0.007),
)

forearm_spheres_rad = (
    0.05,
    0.05,
    0.05,
)


# ============================================================
# LINK 4: WRIST 1
# ============================================================

wrist_1_spheres_pos = (
    (0.0, -0.01, -0.127),
)

wrist_1_spheres_rad = (
    0.05,
)


# ============================================================
# LINK 5: WRIST 2
# ============================================================

wrist_2_spheres_pos = (
    (0.0, 0.0, -0.0997),
)

wrist_2_spheres_rad = (
    0.05,
)


# ============================================================
# LINK 6: WRIST 3
# ============================================================

wrist_3_spheres_pos = (
    (0.0, 0.0, -0.05),
    (0.0, 0.0, -0.12),
)

wrist_3_spheres_rad = (
    0.05,
    0.05,
)


# ============================================================
# COMBINED UR5e COLLISION DATA
# ============================================================

position_list = (
    shoulder_spheres_pos,
    upper_arm_spheres_pos,
    forearm_spheres_pos,
    wrist_1_spheres_pos,
    wrist_2_spheres_pos,
    wrist_3_spheres_pos,
)

radii_list = (
    shoulder_spheres_rad,
    upper_arm_spheres_rad,
    forearm_spheres_rad,
    wrist_1_spheres_rad,
    wrist_2_spheres_rad,
    wrist_3_spheres_rad,
)


ur5e_collision_data = {
    "positions": position_list,
    "radii": radii_list,
}

# Optional: Add data for flange or tool0 if needed for finer collision checking at TCP
# flange_spheres_pos = ([0.0, 0.0, 0.0],)
# flange_spheres_rad = (0.05,)
# ur5e_collision_data["positions"] += (flange_spheres_pos,)
# ur5e_collision_data["radii"] += (flange_spheres_rad,)