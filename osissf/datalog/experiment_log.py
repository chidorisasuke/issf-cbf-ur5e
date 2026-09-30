import os
import pickle

import numpy as np


class ExperimentLogger:
    """
    Logger untuk satu eksperimen UR5e.

    Logger hanya bertanggung jawab terhadap:
    - menampung data per timestep
    - mengubah list menjadi numpy array
    - menyimpan metadata
    - menyimpan hasil sebagai .pkl

    Logger tidak melakukan:
    - control
    - CBF
    - trajectory generation
    - robot communication
    """

    def __init__(
        self,
        save_dir,
        timestamp=None,
    ):
        self.save_dir = save_dir

        os.makedirs(
            self.save_dir,
            exist_ok=True,
        )

        self.timestamp = timestamp

        self.data = {
            "log_t": [],
            "log_x_act": [],
            "log_x_des": [],

            "tau_task": [],
            "q_ddot": [],
            "rhs": [],
            "pa_ddot": [],
            "q_dot": [],
            "pa_dot": [],

            "q_act": [],
            "q_des": [],
            "dq_act": [],
            "dq_task": [],
            "dq_des": [],

            "tau_total": [],
            "tau_task_only": [],
            "tau_null": [],
            "tau_coriolis_grav": [],

            "log_dq_null": [],
            "log_q_null": [],

            "log_v_tcp_speed": [],
            "log_v_tcp_calc": [],

            "log_q_pybullet": [],
            "log_q_ur5e": [],
            "log_x_pybullet_ee": [],
            "log_q_pybullet_pred": [],
            "log_twin_residual": [],

            "log_manipulability": [],

            "log_dq_cmd_safe": [],
            "log_dq_cmd": [],
            "log_dq_cmd_to_send": [],

            "log_h_values": [],
            "log_disturbance": [],

            "log_cbf_time": [],
        }

        self.metadata = {}

    # =========================================================
    # METADATA
    # =========================================================

    def set_metadata(
        self,
        **kwargs,
    ):
        """
        Menambahkan metadata eksperimen.

        Contoh:
            logger.set_metadata(
                fitur="admittance",
                traj="circle_3d",
                cbf="both",
            )
        """
        self.metadata.update(kwargs)

    # =========================================================
    # GENERIC APPEND
    # =========================================================

    def append(
        self,
        name,
        value,
    ):
        """
        Append satu sample ke field log tertentu.
        """
        if name not in self.data:
            raise KeyError(
                f"Unknown log field: {name}"
            )

        if isinstance(
            value,
            np.ndarray,
        ):
            value = value.copy()

        self.data[name].append(value)

    # =========================================================
    # TIMESTEP LOGGING
    # =========================================================

    def log_step(
        self,
        t,
        *,
        x_act=None,
        x_des=None,
        tau_task=None,
        q_ddot=None,
        rhs=None,
        pa_ddot=None,
        q_dot=None,
        pa_dot=None,
        q_act=None,
        q_des=None,
        dq_act=None,
        dq_des=None,
        dq_task=None,
        tau_total=None,
        tau_task_only=None,
        tau_null=None,
        tau_coriolis_grav=None,
        q_null=None,
        dq_null=None,
        v_tcp_speed=None,
        v_tcp_calc=None,
        q_pybullet=None,
        q_ur5e=None,
        x_pybullet_ee=None,
        q_pybullet_pred=None,
        twin_residual=None,
        manipulability=None,
        dq_cmd_safe=None,
        dq_cmd=None,
        dq_cmd_to_send=None,
        h_values=None,
        disturbance=None,
        cbf_time=None,
    ):
        """
        Logging satu timestep.

        None berarti data tersebut tidak tersedia
        untuk mode kontrol yang sedang digunakan.
        """

        self.append(
            "log_t",
            float(t),
        )

        values = {
            "log_x_act": x_act,
            "log_x_des": x_des,

            "tau_task": tau_task,
            "q_ddot": q_ddot,
            "rhs": rhs,
            "pa_ddot": pa_ddot,
            "q_dot": q_dot,
            "pa_dot": pa_dot,

            "q_act": q_act,
            "q_des": q_des,
            "dq_act": dq_act,
            "dq_des": dq_des,
            "dq_task": dq_task,

            "tau_total": tau_total,
            "tau_task_only": tau_task_only,
            "tau_null": tau_null,
            "tau_coriolis_grav": tau_coriolis_grav,

            "log_q_null": q_null,
            "log_dq_null": dq_null,

            "log_v_tcp_speed": v_tcp_speed,
            "log_v_tcp_calc": v_tcp_calc,

            "log_q_pybullet": q_pybullet,
            "log_q_ur5e": q_ur5e,
            "log_x_pybullet_ee": x_pybullet_ee,
            "log_q_pybullet_pred": q_pybullet_pred,
            "log_twin_residual": twin_residual,

            "log_manipulability": manipulability,

            "log_dq_cmd_safe": dq_cmd_safe,
            "log_dq_cmd": dq_cmd,
            "log_dq_cmd_to_send": dq_cmd_to_send,
            "log_h_values": h_values,
            "log_disturbance": disturbance,

            "log_cbf_time": cbf_time,
        }

        for name, value in values.items():

            if value is not None:

                if isinstance(
                    value,
                    np.ndarray,
                ):
                    value = value.copy()

                self.data[name].append(
                    value
                )

    # =========================================================
    # CONVERT TO NUMPY
    # =========================================================

    @staticmethod
    def _to_numpy(
        values,
    ):
        """
        Mengubah list menjadi numpy array.

        Menangani empty list.
        """
        if len(values) == 0:
            return np.array([])

        return np.asarray(values)

    def finalize(
        self,
    ):
        """
        Mengubah seluruh log menjadi numpy arrays.
        """

        output = {}

        for name, values in self.data.items():

            if name == "log_dq_cmd_safe":

                # Hil14 membuang empty arrays untuk field ini.
                filtered = [
                    value
                    for value in values
                    if np.asarray(value).size > 0
                ]

                output[name] = (
                    np.asarray(filtered)
                    if filtered
                    else np.array([])
                )

            else:

                output[name] = self._to_numpy(
                    values
                )

        output.update(
            self.metadata
        )

        return output

    # =========================================================
    # SAVE PICKLE
    # =========================================================

    def save_pickle(
        self,
        filename=None,
    ):
        """
        Simpan seluruh experiment log sebagai .pkl.
        """

        log_data = self.finalize()

        if filename is None:

            if self.timestamp is None:
                filename = "HIL_log.pkl"
            else:
                filename = (
                    f"HIL_log_{self.timestamp}.pkl"
                )

        path = os.path.join(
            self.save_dir,
            filename,
        )

        with open(
            path,
            "wb",
        ) as file:

            pickle.dump(
                log_data,
                file,
            )

        print(
            f"💾 Log data disimpan: {path}"
        )

        return path