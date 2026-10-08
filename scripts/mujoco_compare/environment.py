"""MuJoCo quadrotor plant, observations and shared geometric action prior."""
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "assets/mujoco/charcoal_drone.xml"
DT = .05
HISTORY = 10


def rotor_positions():
    # Front-left, rear-left, rear-right, front-right; +X is the nose.
    return (.24/math.sqrt(2))*np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])


@dataclass(frozen=True)
class Domain:
    mass: float = 1.3
    lag: float = .04
    thrust: float = 1.
    inertia_scale: float = 1.
    drag: float = .25

    def json(self):
        return asdict(self)


def sample_domain(seed, split="train"):
    rng = np.random.default_rng(seed)
    # Hold out the heavy + slow combination, while observing each axis alone.
    while True:
        mass, lag = rng.uniform(.85, 1.65), rng.uniform(.015, .095)
        combined = mass > 1.40 and lag > .065
        if combined == (split == "combination"):
            return Domain(mass, lag, rng.uniform(.85, 1.15),
                          rng.uniform(.8, 1.2), rng.uniform(.15, .4))


def rotation_from_up(up, heading=None):
    z = up / np.linalg.norm(up)
    heading = np.array([1., 0., 0.]) if heading is None else np.asarray(heading)
    y = np.cross(z, heading); y /= np.linalg.norm(y)
    return np.column_stack((np.cross(y, z), y, z))


def trajectory(t, kind="figure8", radius=1.7, period=10., height=1.65):
    w, s = 2 * math.pi / period, np.zeros(18)
    q = w * t
    if kind == "circle":
        s[:3] = [radius * math.cos(q), radius * math.sin(q), height]
        s[3:6] = [-radius*w*math.sin(q), radius*w*math.cos(q), 0]
        acc = np.array([-radius*w*w*math.cos(q), -radius*w*w*math.sin(q), 0])
    elif kind == "figure8":
        s[:3] = [radius*math.sin(q), .5*radius*math.sin(2*q), height + .15*math.sin(q)]
        s[3:6] = [radius*w*math.cos(q), radius*w*math.cos(2*q), .15*w*math.cos(q)]
        acc = np.array([-radius*w*w*math.sin(q), -2*radius*w*w*math.sin(2*q), -.15*w*w*math.sin(q)])
    else:
        s[2], acc = height, np.zeros(3)
    return s, acc


def reference(t, kind="figure8", radius=1.7, period=10., height=1.65,
              heading="fixed", yaw_rate=.45):
    def attitude(at):
        state, acceleration = trajectory(at, kind, radius, period, height)
        if heading == "tangent" and np.linalg.norm(state[3:5]) > 1e-6:
            yaw = math.atan2(state[4], state[3])
        elif heading == "sweep":
            yaw = yaw_rate*at
        else:
            yaw = 0.
        rotation = rotation_from_up(acceleration+[0, 0, 9.81], [math.cos(yaw), math.sin(yaw), 0.])
        return state, acceleration, rotation
    s, acc, R = attitude(t)
    s[6:15] = R.ravel()
    if heading != "fixed":
        # Differentiate SO(3), avoiding discontinuities at atan2's +/- pi wrap.
        derivative = (attitude(t+.001)[2]-attitude(t-.001)[2])/.002
        skew = R.T @ derivative
        s[15:18] = .5*np.array([skew[2, 1]-skew[1, 2], skew[0, 2]-skew[2, 0], skew[1, 0]-skew[0, 1]])
    return s, acc


def allocation(total, torque, trim):
    # Only observable hover calibration is provided; hidden plant mass, lag,
    # inertia and thrust coefficients are unavailable to the controller.
    thrust_scale = 1.3 * 9.81 / (4 * trim)
    arm, yaw = .24 * thrust_scale, .025 * thrust_scale
    tx, ty, tz = torque
    lever = arm/math.sqrt(2)
    return np.clip(np.array([total/4+tx/(4*lever)-ty/(4*lever)+tz/(4*yaw),
                            total/4+tx/(4*lever)+ty/(4*lever)-tz/(4*yaw),
                            total/4-tx/(4*lever)+ty/(4*lever)+tz/(4*yaw),
                            total/4-tx/(4*lever)-ty/(4*lever)-tz/(4*yaw)]), 0, 12.)


def geometric(state, target, acceleration, trim):
    R = state[6:15].reshape(3, 3)
    acc = acceleration + 3.5*(target[:3]-state[:3]) + 2.6*(target[3:6]-state[3:6]) + [0, 0, 9.81]
    target_rotation = target[6:15].reshape(3, 3)
    heading = np.array([target_rotation[1, 1], -target_rotation[0, 1], 0.])
    desired = rotation_from_up(acc, heading)
    error_matrix = .5*(desired.T @ R - R.T @ desired)
    error = np.array([error_matrix[2, 1], error_matrix[0, 2], error_matrix[1, 0]])
    desired_rates = R.T @ target_rotation @ target[15:18]
    torque = -np.array([.021, .023, .040])*(25*error + 8*(state[15:18]-desired_rates))
    return allocation(4*trim*max(0., np.dot(acc, R[:, 2]))/9.81, torque, trim)


def prior(state, time, trim, horizon=15, kind="figure8", radius=1.7, period=10., heading="fixed"):
    actions, targets = [], []
    for j in range(horizon):
        target, acc = reference(time+j*DT, kind, radius, period, heading=heading)
        actions.append(geometric(state if j == 0 else target, target, acc, trim))
        targets.append(reference(time+(j+1)*DT, kind, radius, period, heading=heading)[0])
    return np.asarray(actions, np.float32), np.asarray(targets, np.float32)


def propeller_quaternion(index, time=0.):
    """Illustrative blade phase, not simulated RPM. Viewed from body +Z:

    Adjacent two-blade props park at 90 degrees and counter-rotate. A positive
    body reaction torque corresponds to a clockwise (negative yaw) propeller.
    """
    angle = (index % 2)*math.pi/2 - (1 if index % 2 == 0 else -1)*35*time
    return np.array([math.cos(angle/2), 0., 0., math.sin(angle/2)])


def animate_propellers(plant, time):
    """Update only non-colliding blade visuals; callers hold the viewer lock."""
    for i in range(4):
        plant.model.geom_quat[plant.model.geom(f"prop{i}").id] = propeller_quaternion(i, time)
    mujoco.mj_forward(plant.model, plant.data)


class Plant:
    def __init__(self, domain=Domain()):
        values = (domain.mass, domain.lag, domain.thrust, domain.inertia_scale)
        if not all(np.isfinite(v) and v > 0 for v in values) or not np.isfinite(domain.drag) or domain.drag < 0:
            raise ValueError("Mass, motor lag, thrust gain and inertia scale must be positive; drag must be nonnegative")
        self.domain = domain
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self.model.body_mass[self.body] = domain.mass
        self.model.body_inertia[self.body] = np.array([.021, .023, .040])*domain.inertia_scale
        self.model.actuator_dynprm[:, 0] = domain.lag
        self.model.actuator_gainprm[:, 0] = domain.thrust
        self.data = mujoco.MjData(self.model)
        mujoco.mj_setConst(self.model, self.data)
        self.trim = domain.mass*9.81/(4*domain.thrust)
        self.reset()

    def reset(self, state=None):
        mujoco.mj_resetData(self.model, self.data)
        if state is None:
            state = reference(0, "hover")[0]
        self.data.qpos[:3] = state[:3]
        mujoco.mju_mat2Quat(self.data.qpos[3:7], np.ascontiguousarray(state[6:15]))
        self.data.qvel[:3] = state[3:6]
        self.data.qvel[3:6] = state[15:18]
        self.data.act[:] = self.trim
        self.data.ctrl[:] = self.trim
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        return np.concatenate((self.data.qpos[:3], self.data.qvel[:3],
                               self.data.xmat[self.body], self.data.qvel[3:6])).astype(np.float32)

    def snapshot(self):
        return np.concatenate((self.data.qpos, self.data.qvel, self.data.act, [self.data.time])).copy()

    def restore(self, snapshot):
        self.data.qpos[:] = snapshot[:7]
        self.data.qvel[:] = snapshot[7:13]
        self.data.act[:] = snapshot[13:17]
        self.data.time = snapshot[17]
        self.data.qacc_warmstart[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, 0, 12)
        contact = False
        for _ in range(round(DT/self.model.opt.timestep)):
            self.data.xfrc_applied[self.body, :3] = -self.domain.drag*self.data.qvel[:3]
            mujoco.mj_step(self.model, self.data)
            contact |= self.data.ncon > 0
        mujoco.mj_forward(self.model, self.data)
        return self.state(), contact


def make_scene():
    """Original procedural asset: no downloaded mesh or license dependency."""
    parts = []
    # Body frame: +X nose, +Y left, +Z up. The physical thrust sites and visible
    # arms share positions; allocation() uses the corresponding moment arms.
    for i, (x, y) in enumerate(rotor_positions()):
        quat = ' '.join(str(v) for v in propeller_quaternion(i))
        parts += [f'<geom type="capsule" fromto="0 0 .02 {x} {y} .035" size=".021" material="carbon"/>',
                  f'<geom type="cylinder" pos="{x} {y} .043" size=".038 .025" material="metal"/>',
                  f'<geom type="cylinder" pos="{x} {y} .071" size=".029 .007" material="accent"/>',
                  f'<geom name="prop{i}" type="ellipsoid" pos="{x} {y} .082" quat="{quat}" size=".118 .016 .003" material="prop" contype="0" conaffinity="0"/>',
                  f'<geom name="rotor_disc{i}" type="cylinder" pos="{x} {y} .082" size=".116 .001" rgba=".25 .29 .31 .13" contype="0" conaffinity="0"/>',
                  f'<geom type="sphere" pos="{x} {y} .089" size=".012" material="metal"/>',
                  f'<site name="rotor{i}" pos="{x} {y} .082" size=".003" rgba="0 0 0 0"/>']
    rails = []
    for side, y in enumerate((-.105, .105)):
        rails += [f'<geom name="skid{side}" type="capsule" fromto="-.14 {y} -.13 .14 {y} -.13" size=".009" material="metal"/>']
        for x in (-.075, .075):
            rails += [f'<geom type="capsule" fromto="{x} {y*.6} -.025 {x} {y} -.13" size=".009" material="carbon"/>']
    floor = []
    for i in range(-12, 13):
        for axis in (0, 1):
            pos = f'{i} 0 .001' if axis == 0 else f'0 {i} .001'
            size = '.004 12 .001' if axis == 0 else '12 .004 .001'
            floor.append(f'<geom type="box" pos="{pos}" size="{size}" rgba=".115 .13 .14 1" contype="0" conaffinity="0"/>')
    for x in (-6, 6):
        floor.append(f'<geom type="box" pos="{x} 0 .006" size=".018 6 .005" material="accent" contype="0" conaffinity="0"/>')
    # Calibration aids are visual-only; they cannot secretly alter benchmark
    # contacts. The physical task remains flight above a single ground plane.
    for x in (-3.5, 3.5):
        for y in (-3.5, 3.5):
            floor.append(f'<geom type="cylinder" pos="{x} {y} .007" size=".52 .006" rgba=".12 .15 .17 1" contype="0" conaffinity="0"/>')
            for a in np.linspace(0, 2*math.pi, 49)[:-1]:
                b = a+2*math.pi/48
                p = f'{x+.43*math.cos(a)} {y+.43*math.sin(a)} .016'
                q = f'{x+.43*math.cos(b)} {y+.43*math.sin(b)} .016'
                floor.append(f'<geom type="capsule" fromto="{p} {q}" size=".008" rgba=".40 .48 .50 1" contype="0" conaffinity="0"/>')
            for dx in (-.13, .13):
                floor.append(f'<geom type="box" pos="{x+dx} {y} .018" size=".018 .18 .002" rgba=".56 .64 .65 1" contype="0" conaffinity="0"/>')
            floor.append(f'<geom type="box" pos="{x} {y} .018" size=".13 .018 .002" rgba=".56 .64 .65 1" contype="0" conaffinity="0"/>')
    for x in (-3.0, 0, 3.0):
        for a in np.linspace(0, 2*math.pi, 65)[:-1]:
            b = a+2*math.pi/64
            p = f'{x+.72*math.cos(a)} 3.0 {1.65+.72*math.sin(a)}'
            q = f'{x+.72*math.cos(b)} 3.0 {1.65+.72*math.sin(b)}'
            floor.append(f'<geom type="capsule" fromto="{p} {q}" size=".012" rgba=".20 .35 .37 1" contype="0" conaffinity="0"/>')
        floor.append(f'<geom type="capsule" fromto="{x} 3.0 .02 {x} 3.0 .91" size=".018" material="metal" contype="0" conaffinity="0"/>')
    for i in range(-5, 6):
        floor.append(f'<geom type="box" pos="{i} -4.8 .009" size=".28 .025 .006" rgba=".65 .47 .20 1" contype="0" conaffinity="0"/>')
    actuators = [f'<general name="motor{i}" site="rotor{i}" gear="0 0 1 0 0 {(.025 if i%2 == 0 else -.025)}" dyntype="filterexact" dynprm=".04" gainprm="1" ctrllimited="true" ctrlrange="0 12"/>' for i in range(4)]
    xml = '''<mujoco model="JEPA Gym / X">
  <compiler angle="radian" autolimits="true"/>
  <option timestep=".0025" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096" offsamples="4"/>
    <headlight ambient=".22 .24 .28" diffuse=".55 .58 .65" specular=".35 .35 .35"/>
    <map znear=".01" zfar="80" shadowclip="12"/><rgba haze=".055 .065 .08 1"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".035 .045 .065" rgb2=".10 .12 .15" width="512" height="3072"/>
    <material name="floor" rgba=".055 .064 .075 1" specular=".3" shininess=".45" reflectance=".12"/>
    <material name="carbon" rgba=".09 .105 .125 1" specular=".65" shininess=".65"/>
    <material name="shell" rgba=".24 .28 .31 1" specular=".75" shininess=".7"/>
    <material name="metal" rgba=".38 .43 .46 1" specular=".9" shininess=".85"/>
    <material name="accent" rgba=".13 .75 .68 1" emission=".35"/>
    <material name="prop" rgba=".10 .13 .15 1" specular=".6" shininess=".7"/>
    <material name="glass" rgba=".02 .035 .05 1" specular="1" shininess="1"/>
  </asset>
  <default><geom friction=".8 .01 .001" condim="3"/></default>
  <worldbody>
    <light pos="-3 -4 8" dir=".3 .4 -1" diffuse=".9 .88 .82" specular=".8 .8 .8" castshadow="true"/>
    <light pos="3 4 6" dir="-.4 -.5 -1" diffuse=".35 .5 .65" castshadow="false"/>
    <geom name="floor" type="plane" size="20 20 .1" material="floor"/>
    FLOOR
    <camera name="overview" pos="6 -8 5.3" xyaxes=".8 .6 0 -.27 .36 .9" fovy="42"/>
    <body name="drone" pos="0 0 1.65">
      <freejoint/>
      <inertial pos="0 0 0" mass="1.3" diaginertia=".021 .023 .040"/>
      <geom type="ellipsoid" size=".14 .09 .055" material="shell"/>
      <geom type="box" pos="-.015 0 .052" size=".072 .049 .014" material="carbon"/>
      <geom name="nose_mark_left" type="capsule" fromto=".025 .022 .070 .063 0 .070" size=".004" material="accent" contype="0" conaffinity="0"/>
      <geom name="nose_mark_right" type="capsule" fromto=".025 -.022 .070 .063 0 .070" size=".004" material="accent" contype="0" conaffinity="0"/>
      <geom type="ellipsoid" pos=".105 0 -.015" size=".045 .046 .03" material="carbon"/>
      <geom type="cylinder" pos=".141 0 -.016" size=".023 .012" quat=".7071068 0 .7071068 0" material="metal"/>
      <geom name="camera_lens" type="cylinder" pos=".155 0 -.016" size=".019 .002" quat=".7071068 0 .7071068 0" material="glass"/>
      <geom type="capsule" fromto="-.08 0 .04 -.11 0 .12" size=".004" material="metal"/>
      PARTS
      RAILS
    </body>
  </worldbody>
  <actuator>ACTUATORS</actuator>
</mujoco>'''
    xml = xml.replace('FLOOR', '\n'.join(floor)).replace('PARTS', '\n'.join(parts)).replace('RAILS', '\n'.join(rails)).replace('ACTUATORS', '\n'.join(actuators))
    SCENE.parent.mkdir(parents=True, exist_ok=True)
    SCENE.write_text(xml)


if __name__ == "__main__":
    make_scene()
