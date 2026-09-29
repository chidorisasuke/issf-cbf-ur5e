import numpy as np


def compute_tracking_metrics(
    x_des,
    x_act,
    dq_cmd=None,
    dq_cmd_safe=None,
):
    """
    Menghitung metrik tracking Cartesian.

    Returns:
        {
            "rmse_total": float,
            "rmse_free": float,
            "max_error": float,
        }
    """

    x_des = np.asarray(
        x_des,
        dtype=float,
    )

    x_act = np.asarray(
        x_act,
        dtype=float,
    )

    if x_des.size == 0 or x_act.size == 0:
        return {
            "rmse_total": 0.0,
            "rmse_free": 0.0,
            "max_error": 0.0,
        }

    error = (
        x_des - x_act
    )

    # ----------------------------------------------
    # RMSE total
    # ----------------------------------------------

    rmse_total = np.sqrt(
        np.mean(
            error**2
        )
    )

    # ----------------------------------------------
    # Maximum absolute error
    # ----------------------------------------------

    max_error = np.max(
        np.abs(error)
    )

    # ----------------------------------------------
    # Free-space RMSE
    # ----------------------------------------------

    if (
        dq_cmd is not None
        and dq_cmd_safe is not None
    ):

        dq_cmd = np.asarray(
            dq_cmd,
            dtype=float,
        )

        dq_cmd_safe = np.asarray(
            dq_cmd_safe,
            dtype=float,
        )

        if (
            len(dq_cmd)
            == len(dq_cmd_safe)
        ):

            diff_vel = np.linalg.norm(
                dq_cmd
                - dq_cmd_safe,
                axis=1,
            )

            is_free_space = (
                diff_vel < 1e-3
            )

            if np.any(
                is_free_space
            ):

                error_free = (
                    error[
                        is_free_space
                    ]
                )

                rmse_free = (
                    np.sqrt(
                        np.mean(
                            error_free**2
                        )
                    )
                )

            else:

                rmse_free = 0.0

        else:

            rmse_free = rmse_total

    else:

        rmse_free = rmse_total

    return {
        "rmse_total": float(
            rmse_total
        ),
        "rmse_free": float(
            rmse_free
        ),
        "max_error": float(
            max_error
        ),
    }


def compute_cbf_metrics(
    dq_cmd,
    dq_cmd_safe,
):
    """
    Menghitung besar intervensi CBF.
    """

    dq_cmd = np.asarray(
        dq_cmd,
        dtype=float,
    )

    dq_cmd_safe = np.asarray(
        dq_cmd_safe,
        dtype=float,
    )

    if (
        dq_cmd.size == 0
        or dq_cmd_safe.size == 0
    ):
        return {
            "max_deviation": 0.0,
            "avg_deviation": 0.0,
            "intervention_count": 0,
        }

    deviation = np.abs(
        dq_cmd
        - dq_cmd_safe
    )

    max_deviation = np.max(
        deviation
    )

    avg_deviation = np.mean(
        deviation
    )

    intervention_count = np.sum(
        np.linalg.norm(
            dq_cmd
            - dq_cmd_safe,
            axis=1,
        ) > 1e-3
    )

    return {
        "max_deviation": float(
            max_deviation
        ),
        "avg_deviation": float(
            avg_deviation
        ),
        "intervention_count": int(
            intervention_count
        ),
    }


def compute_digital_twin_metrics(
    q_ur5e,
    q_pybullet,
):
    """
    Mengukur error antara state joint UR5e
    dan Digital Twin PyBullet.
    """

    q_ur5e = np.asarray(
        q_ur5e,
        dtype=float,
    )

    q_pybullet = np.asarray(
        q_pybullet,
        dtype=float,
    )

    if (
        q_ur5e.size == 0
        or q_pybullet.size == 0
    ):
        return {
            "avg_joint_difference": 0.0,
            "max_joint_difference": 0.0,
        }

    difference = np.abs(
        q_ur5e
        - q_pybullet
    )

    return {
        "avg_joint_difference": float(
            np.mean(difference)
        ),
        "max_joint_difference": float(
            np.max(difference)
        ),
    }


def compute_all_metrics(
    log_data,
):
    """
    Menghitung seluruh metrik yang dapat
    diperoleh dari experiment log.
    """

    metrics = {}

    # ----------------------------------------------
    # Tracking
    # ----------------------------------------------

    if (
        "log_x_des" in log_data
        and "log_x_act" in log_data
    ):

        dq_cmd = log_data.get(
            "log_dq_cmd"
        )

        dq_cmd_safe = log_data.get(
            "log_dq_cmd_safe"
        )

        metrics["tracking"] = (
            compute_tracking_metrics(
                log_data["log_x_des"],
                log_data["log_x_act"],
                dq_cmd=dq_cmd,
                dq_cmd_safe=dq_cmd_safe,
            )
        )

    # ----------------------------------------------
    # CBF
    # ----------------------------------------------

    if (
        "log_dq_cmd" in log_data
        and "log_dq_cmd_safe" in log_data
    ):

        if (
            log_data["log_dq_cmd_safe"].size
            > 0
        ):

            metrics["cbf"] = (
                compute_cbf_metrics(
                    log_data[
                        "log_dq_cmd"
                    ],
                    log_data[
                        "log_dq_cmd_safe"
                    ],
                )
            )

    # ----------------------------------------------
    # Digital Twin
    # ----------------------------------------------

    if (
        "log_q_ur5e" in log_data
        and "log_q_pybullet" in log_data
    ):

        if (
            log_data["log_q_ur5e"].size > 0
            and
            log_data["log_q_pybullet"].size > 0
        ):

            metrics["digital_twin"] = (
                compute_digital_twin_metrics(
                    log_data[
                        "log_q_ur5e"
                    ],
                    log_data[
                        "log_q_pybullet"
                    ],
                )
            )

    return metrics