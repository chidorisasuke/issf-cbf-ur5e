# ISSf-CBF UR5e

Safety filter **CBF / ISSf-CBF** (*Input-to-State Safe Control Barrier Function*) dan kendali **admittance** ruang tugas untuk robot **UR5e**, dijalankan secara *hardware-in-the-loop* (HIL) melalui RTDE dengan **digital twin** PyBullet.

Repositori ini dikembangkan dari pustaka [Operational Space Control Barrier Functions (OSCBF)](https://github.com/StanfordASL/oscbf) milik Stanford Autonomous Systems Laboratory (lisensi MIT), berbasis *commit* `6082329`. Modifikasi yang dilakukan meliputi:

- penambahan model kolisi UR5e,
- penyesuaian perhitungan inersia gabungan pada modul `oscbf/utils/urdf_parser.py`,
- pengembangan paket `osissf`, yang memuat seluruh implementasi kendali *admittance*, penyaring keselamatan CBF dan ISSf-CBF, *digital twin*, serta antarmuka HIL.

Versi kode yang digunakan dalam skripsi ditandai dengan *tag* `v1.0-skripsi`.

## Struktur Program

| Modul | Fungsi |
|---|---|
| `osissf/config/` | Parameter eksperimen dan parameter kendali |
| `osissf/core/` | Kendali *admittance*, dinamika Pinocchio, *null-space*, serta CBF dan ISSf-CBF |
| `osissf/robot/` | Komunikasi UR5e melalui protokol RTDE |
| `osissf/simulation/` | Lingkungan PyBullet dan sinkronisasi *digital twin* |
| `osissf/trajectory/` | Pembangkit lintasan referensi |
| `osissf/datalog/` | Perekaman data dan perhitungan metrik |
| `osissf/visualization/` | Visualisasi PyBullet dan penggambaran grafik hasil |
| `osissf/monitoring/` | Pemantauan waktu nyata melalui Prometheus |
| `osissf/experiment/` | Program utama `run_ur5e.py` |

Alur perintah kendali pada setiap siklus (125 Hz):

```text
trajectory → task controller / admittance → dq_task
dq_des = dq_task + dq_null                  (nominal, sebelum safety filter)
dq_des → CBF / ISSf-CBF → dq_cmd_safe → dq_cmd_to_send → UR5e speedJ()
```

## Kebutuhan Perangkat Lunak

Diuji pada Ubuntu 22.04 dengan Python 3.10.

| Pustaka | Versi |
|---|---|
| JAX / JAXlib | 0.4.30 |
| CBFpy | 0.0.1 |
| Pinocchio | 3.4.0 |
| PyBullet | 3.2.7 |
| ur_rtde | 1.6.2 |
| NumPy | 1.26.4 |
| prometheus-client | 0.23.1 |

Catatan versi:

- **JAX** dikunci pada 0.4.30 karena versi setelahnya mengalami penurunan kinerja komputasi pada CPU.
- **CBFpy** dikunci pada 0.0.1 karena ada perubahan API pada versi berikutnya.
- **Pinocchio** dipasang dari `conda-forge`, bukan dari `pip`: paket `pin` di PyPI mensyaratkan NumPy 2, sedangkan CBFpy mensyaratkan NumPy < 2, sehingga keduanya tidak dapat dipenuhi bersamaan melalui `pip`.

## Instalasi

```bash
conda create -n oscbf-env python=3.10
conda activate oscbf-env
conda install -c conda-forge pinocchio=3.4.0
git clone https://github.com/chidorisasuke/issf-cbf-ur5e
cd issf-cbf-ur5e
pip install -e .
```

Prosedur ini telah diverifikasi pada kontainer Docker bersih, sehingga tidak bergantung pada berkas atau konfigurasi lokal.

## Menjalankan Program

### 1. Jalankan URSim

Simulator URSim e-Series 5.24 dijalankan melalui Docker:

```bash
docker run -it --name my_ursim_ready \
    -p 5900:5900 -p 29999:29999 -p 30001:30001 \
    -p 30002:30002 -p 30003:30003 -p 30004:30004 \
    universalrobots/ursim_e-series:5.24
```

Buka PolyScope (VNC di port 5900), nyalakan robot, lalu lepas rem.

### 2. Jalankan eksperimen

```bash
osissf-run --ip 127.0.0.1 --mode real --traj circle_3d \
           --fitur admittance --cbf both --issf
```

Perintah `osissf-run` setara dengan `python -m osissf.experiment.run_ur5e`.

Pada mode `real`, robot terlebih dahulu di-*homing*, lalu digerakkan ke titik awal lintasan (IK + `moveJ`), ditunggu hingga diam, baru kemudian loop kendali dimulai. PyBullet berfungsi sebagai *digital twin* yang mengikuti state aktual robot.

Untuk simulasi murni tanpa URSim/robot, gunakan `--mode sim` (PyBullet menjadi *plant*).

### Robot UR5e fisik

Isi `--ip` dengan alamat IP pengendali robot. **Sebelum menjalankan**, atur konfigurasi keselamatan pada *teach pendant* (batas kecepatan dan bidang keselamatan) sebagai lapisan pengaman yang bekerja independen dari program ini.

## Daftar Argumen

| Argumen | Pilihan (bawaan) | Keterangan |
|---|---|---|
| `--ip` | alamat IP (`127.0.0.1`) | Alamat pengendali robot; `127.0.0.1` untuk URSim |
| `--mode` | `sim`, `real` (`real`) | `real`: URSim atau robot fisik melalui RTDE; `sim`: PyBullet sebagai *plant* |
| `--traj` | `linear_x`, `linear_y`, `circle`, `circle_3d`, `point`, `mouse`, `wp`, `pap3`, `obstacles` (`linear_x`) | Lintasan referensi |
| `--fitur` | `torque`, `admittance`, `JSA`, `TSA` (`admittance`) | Strategi kendali nominal |
| `--speed_mode` | `slow`, `normal`, `fast` (`normal`) | Kecepatan lintasan dan nilai penguatan kendali |
| `--cbf` | `off`, `containment`, `obstacle`, `both` (`both`) | Jenis kendala keselamatan yang aktif |
| `--issf` | *flag* (nonaktif) | Aktifkan ISSf-CBF; tanpa flag ini digunakan CBF standar |
| `--obstacle` | `sphere`, `tube` (`sphere`) | Bentuk geometri halangan |
| `--show` | `orange`, `blue`, `both` (`orange`) | Halangan yang ditampilkan pada visualisasi |
| `--nullspace` | `on`, `off` (`on`) | Kendali *null-space* untuk menjaga postur |
| `--speed` | `act`, `calc` (`calc`) | Sumber kecepatan Cartesian; `calc` = J·q̇, `act` = dibaca dari TCP |
| `--command` | `speedJ`, `moveL`, `servoL` (`speedJ`) | Jenis perintah gerak yang dikirim melalui RTDE |
| `--dist` | `on`, `off` (`off`) | Tambahkan gangguan buatan pada sinyal kendali |
| `--Kp`, `--Kd` | bilangan riil (mengikuti `--speed_mode`) | Penguatan proporsional dan derivatif ruang tugas |
| `--Kpn`, `--kdn` | bilangan riil (mengikuti `--nullspace`) | Penguatan proporsional dan derivatif *null-space* |
| `--Kj` | bilangan riil (`50.0`) | Penguatan untuk strategi JSA dan TSA |
| `--loop` | *flag* (nonaktif) | Jalankan lintasan berulang tanpa batas waktu |
| `--setup-only` | *flag* (nonaktif) | Hanya *homing* dan bergerak ke titik awal lintasan |
| `--show_env` | `on`, `off` (`on`) | Tampilkan lingkungan dan kotak batas di PyBullet |
| `--show_actual` | `on`, `off` (`off`) | Tampilkan posisi aktual *end-effector* robot |
| `--cek` | *flag* (nonaktif) | Pemeriksaan lingkungan dan lintasan |
| `--save-log` | *flag* (nonaktif) | Simpan log ke berkas `.pkl` |
| `--record` | *flag* (nonaktif) | Rekam simulasi menjadi MP4 |
| `--pertimbangan` | *flag* (nonaktif) | Opsi lama (*legacy*) |

Daftar lengkap juga dapat dilihat dengan `osissf-run --help`.

## Lisensi dan Sitasi

Dirilis di bawah lisensi MIT (lihat `LICENSE`). Jika menggunakan kode ini, mohon sitasi juga karya OSCBF asli:

```bibtex
@inproceedings{morton2025oscbf,
  author={Morton, Daniel and Pavone, Marco},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  title={Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions},
  year={2025},
  pages={187-194},
  doi={10.1109/IROS60139.2025.11246389}
}
```

---

## Upstream: Operational Space Control Barrier Functions

Bagian berikut adalah README asli dari [StanfordASL/oscbf](https://github.com/StanfordASL/oscbf).

Code for *"Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions"* -- Daniel Morton and Marco Pavone

Accepted to IROS 2025, Hangzhou

[![Paper](http://img.shields.io/badge/arXiv-2503.06736-B31B1B.svg)](https://arxiv.org/abs/2503.06736)

### What is OSCBF?

This is a safe, high-performance, and easy-to-use controller / safety filter for robotic manipulators.

With OSCBF, you can...
- Operate at kilohertz speed even with over 400 active safety constraints
- Design safety constraints (barrier functions) easily via CBFpy
- Enforce safety on both torque-controlled and velocity-controlled robots
- Either apply a safety filter on top of your existing controller, or use our provided controller

In general, this will be especialy useful for enforcing safety during **teleoperation** or while executing **learned policies**

For more details and videos, check out the [project webpage](https://stanfordasl.github.io/oscbf/) as well as the [CBFpy documentation](https://danielpmorton.github.io/cbfpy/).


### How do I use this?

Check out the `examples/` folder for interactive demos in Pybullet! This is the best place to start

If you're applying this to a different robot, you'll need to provide a URDF -- we can parse the kinematics and dynamics from that. Some notes:
- I haven't written an MJCF parser yet, but it should be feasible
- All joints that shouldn't be controlled as part of the kinematic chain should be set to `fixed` (gripper joints, for instance). Since you probably want to still be able to use the gripper joints in simulation, the best way to handle this is to make a copy of the URDF: load the non-fixed one in sim, and parse the fixed one with OSCBF
- I manually defined the collision model for the Franka. There are probably better ways to parse or generate this data from meshes, but I haven't done it yet. For now, I'd recommend doing the same for your robot.

### FAQ

- "I already have a well-tuned operational space controller! I don't want to replace that"
  - You can use OSCBF as a safety filter on top of your existing controller!
- "I'm working with a mobile manipulator. Does this still work?"
  - Yes! We support prismatic and revolute joints, so adding a mobile base just adds 3DOF (PPR) to the beginning of the kinematic chain.
- "I'm controlling joint-space motions -- does this still apply?"
  - Depending on the task, you will likely want to modify the objective function. But, the robot dynamics and CBF formulation will still be useful!


### Installation

A virtual environment is optional, but highly recommended. For `pyenv` installation instructions, see [here](https://danielpmorton.github.io/cbfpy/pyenv).

```
git clone https://github.com/stanfordasl/oscbf
cd oscbf
pip install -e .
```

Note: This code will work with most versions of `jax`, but there seems to have been a CPU slowdown introduced in version `0.4.32`. To avoid this, I use version `0.4.30`, which is the version indicated in this repo's `pyproject.toml` as well. However, feel free to use any version you like.

This has been tested on Python 3.10 and 3.11, on Ubuntu 22.04.


### Documentation

See the CBFpy documentation, available at [this link](https://danielpmorton.github.io/cbfpy)


### Hardware / ROS2 

The code used to run the Franka Panda hardware experiments is available at [this link](https://github.com/StanfordASL/oscbf_hardware_ws)
