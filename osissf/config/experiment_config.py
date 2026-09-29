from dataclasses import dataclass, field
import argparse
import numpy as np

Q_HOME = np.array(
    [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
    dtype=float,
)

@dataclass
class ExperimentConfig:
    ip: str
    save_log: bool
    mode: str
    traj: str
    speed: str
    fitur: str
    speed_mode: str
    nullspace: str
    command: str
    cbf: str
    issf: bool
    obstacle: str
    show: str
    dist: str
    cek: bool
    setup_only: bool
    Kp: float | None
    Kd: float | None
    Kpn: float | None
    kdn: float | None
    Kj: float | None
    show_actual: str
    loop: bool
    pertimbangan: bool
    show_env: str
    record: bool
    q_home: np.ndarray

    # ==========================================================
    # ROBOT / MODEL
    # ==========================================================

    urdf_path: str = "assets/ur5e/ur5e_physics.urdf"
    ee_link_pinocchio: str = "wrist_3_link"

    # ==========================================================
    # TRAJECTORY  (frame RTDE / base UR, satuan meter)
    # ==========================================================
    # Pusat lintasan untuk linear_x, linear_y, circle_3d, point,
    # obstacles. hil14: TRAJECTORY_OFFSET = [-0.53, -0.08, 0.40]
    # Titik awal circle_3d = offset + [A, 0, 0].

    runtime: float = 60.0
    A: float = 0.32          # amplitudo / jari-jari (m)

    trajectory_offset: np.ndarray = field(
        default_factory=lambda: np.array(
            [-0.53, -0.08, 0.40],
            dtype=float,
        )
    )

    # Pusat lintasan khusus --traj circle (hil14: CIRCLE_2D_OFFSET)
    circle_2d_offset: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.425, 0.0, 0.30],
            dtype=float,
        )
    )

    tilt_angle: float = np.pi / 6   # kemiringan bidang circle_3d

    # ==========================================================
    # ENVIRONMENT
    # ==========================================================
    # Box containment EE / whole-body dan posisi obstacle
    # didefinisikan di pybullet_env.py (_create_obstacles ->
    # containment_params), satu sumber untuk visual & CBF (hil14).

    table_height: float = 0.59

    robot_base_position: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.0, 0.0, 0.59],
            dtype=float,
        )
    )

    # ==========================================================
    # SAFETY
    # ==========================================================

    safety_margin: float = 0.00     # hil14: SAFETY_MARGIN = 0.00
    obstacle_margin: float = 0.02

    epsilon_0: float = 0.20
    issf_threshold: float = 0.20

    alpha_c_default: float = 2.0
    alpha_c_circle_3d: float = 8.0

    @property
    def alpha_c(self) -> float:
        if self.traj == "circle_3d":
            return self.alpha_c_circle_3d
        return self.alpha_c_default


def create_parser():
    parser = argparse.ArgumentParser(
        description="UR5e HIL Control: Strategy 2 (Calculator)"
    )

    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="IP Robot (127.0.0.1 untuk URSim)")
    parser.add_argument("--save-log", action="store_true",
                        help="Simpan log ke file .pkl")
    parser.add_argument("--mode", choices=["sim", "real"], default="real",
                        help="Mode operasi: sim atau real")
    parser.add_argument(
        "--traj",
        choices=["linear_x", "linear_y", "circle", "circle_3d", "point",
                 "mouse", "wp", "pap3", "obstacles"],
        default="linear_x",
        help="Pilih trajectory",
    )
    parser.add_argument("--speed", choices=["act", "calc"], default="calc",
                        help="act = TCP speed aktual, calc = Jacobian @ joint velocity")
    parser.add_argument("--fitur", choices=["torque", "admittance", "JSA", "TSA"],
                        default="admittance", help="Fitur kontrol")
    parser.add_argument("--speed_mode", choices=["slow", "normal", "fast"],
                        default="normal", help="Kecepatan gerakan")
    parser.add_argument("--nullspace", choices=["on", "off"], default="on",
                        help="Aktif/nonaktif nullspace")
    parser.add_argument("--command", choices=["speedJ", "moveL", "servoL"],
                        default="speedJ", help="Jenis command robot")
    parser.add_argument("--cbf", choices=["off", "containment", "obstacle", "both"],
                        default="both", help="Mode CBF")
    parser.add_argument("--issf", action="store_true", help="Aktifkan ISSf-CBF")
    parser.add_argument("--obstacle", choices=["sphere", "tube"], default="sphere",
                        help="Tipe obstacle")
    parser.add_argument("--show", choices=["orange", "blue", "both"], default="orange",
                        help="Obstacle yang ditampilkan")
    parser.add_argument("--dist", choices=["on", "off"], default="off",
                        help="Disturbance aktif")
    parser.add_argument("--cek", action="store_true",
                        help="Environment dan trajectory check")
    parser.add_argument("--setup-only", action="store_true",
                        help="Hanya homing dan move ke start")
    parser.add_argument("--Kp", type=float, default=None,
                        help="Task-space proportional gain")
    parser.add_argument("--Kd", type=float, default=None,
                        help="Task-space derivative gain")
    parser.add_argument("--Kpn", type=float, default=None,
                        help="Null-space proportional gain")
    parser.add_argument("--kdn", type=float, default=None,
                        help="Null-space derivative gain")
    parser.add_argument("--Kj", type=float, default=None, help="JSA/TSA gain")
    parser.add_argument("--show_actual", choices=["on", "off"], default="off",
                        help="Tampilkan posisi aktual robot fisik")
    parser.add_argument("--loop", action="store_true",
                        help="Jalankan loop tanpa batas waktu")
    parser.add_argument("--pertimbangan", action="store_true", help="Legacy option")
    parser.add_argument("--show_env", choices=["on", "off"], default="on",
                        help="Tampilkan environment PyBullet")
    parser.add_argument("--record", action="store_true",
                        help="Rekam simulasi menjadi MP4")

    return parser


def parse_experiment_config():
    args = create_parser().parse_args()

    return ExperimentConfig(
        ip=args.ip,
        save_log=args.save_log,
        mode=args.mode,
        traj=args.traj,
        speed=args.speed,
        fitur=args.fitur,
        speed_mode=args.speed_mode,
        nullspace=args.nullspace,
        command=args.command,
        cbf=args.cbf,
        issf=args.issf,
        obstacle=args.obstacle,
        show=args.show,
        dist=args.dist,
        cek=args.cek,
        setup_only=args.setup_only,
        Kp=args.Kp,
        Kd=args.Kd,
        Kpn=args.Kpn,
        kdn=args.kdn,
        Kj=args.Kj,
        show_actual=args.show_actual,
        loop=args.loop,
        pertimbangan=args.pertimbangan,
        show_env=args.show_env,
        record=args.record,
        q_home=Q_HOME.copy(),
    )