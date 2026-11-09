"""MuJoCo 시뮬레이션을 3D 재생용 궤적으로 기록한다.
각 장면: MuJoCo 물체의 pos/quat(실제 시뮬레이션) + 해석해 '고스트' 물체의 pos/quat.
출력: scenes.json (웹 3D 뷰어가 그대로 재생)
"""
import json, math
import numpy as np
import mujoco
from scipy.integrate import solve_ivp

G = 9.81
FPS = 60


def M(xml):
    return mujoco.MjModel.from_xml_string(xml)


def qy(th):  # y축 회전 쿼터니언 (w,x,y,z)
    return [math.cos(th / 2), 0, math.sin(th / 2), 0]


def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return [w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
            w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2]


def rnd(a, n=5):
    return [round(float(v), n) for v in a]


class Rec:
    def __init__(self, n):
        self.tr = [[] for _ in range(n)]
        self.t, self.sim, self.ref, self.err = [], [], [], []

    def add(self, t, poses, sim, ref, err):
        self.t.append(round(t, 4))
        for i, (p, q) in enumerate(poses):
            self.tr[i].extend(rnd(p, 4) + rnd(q, 4))
        self.sim.append(float(sim)); self.ref.append(float(ref)); self.err.append(float(err))


def run(m, d, dur, cb):
    """dur 초 동안 스텝, 1/FPS 마다 cb(t) 호출."""
    every = max(1, round(1 / FPS / m.opt.timestep))
    mujoco.mj_forward(m, d)
    cb(0.0)
    k = 0
    while d.time < dur - 1e-9:
        mujoco.mj_step(m, d); k += 1
        if k % every == 0:
            cb(d.time)


def pose(d, b):
    return d.xpos[b].copy(), d.xquat[b].copy()


SCENES = []


def scene(**kw):
    SCENES.append(kw)


# 1. 자유낙하 ---------------------------------------------------------------
def s_freefall():
    m = M(f"""<mujoco><option timestep="0.001" integrator="RK4" gravity="0 0 -{G}"/>
      <worldbody><body pos="0 0 10"><freejoint/><geom type="sphere" size="0.3" mass="1" contype="0" conaffinity="0"/></body></worldbody></mujoco>""")
    d = mujoco.MjData(m); r = Rec(2)
    def cb(t):
        z = 10 - 0.5 * G * t * t
        r.add(t, [pose(d, 1), ([0, 0, z], [1, 0, 0, 0])], d.xpos[1][2], z, abs(d.xpos[1][2] - z))
    run(m, d, 1.35, cb)
    scene(key="freefall", title="자유낙하", formula="z(t) = z₀ − ½gt²", qty="높이 z (m)",
          bodies=[dict(shape="sphere", size=[0.3], color="sim", trail=True), dict(shape="sphere", size=[0.3], ghost=True)],
          ground=dict(size=4), cam=dict(pos=[4.5, -7, 6.5], target=[0, 0, 4.6]), rec=r)


# 2. 포물선 ------------------------------------------------------------------
def s_projectile():
    v, th = 20.0, math.radians(35)
    m = M(f"""<mujoco><option timestep="0.0005" integrator="RK4" gravity="0 0 -{G}"/>
      <worldbody><body pos="0 0 0.3"><freejoint/><geom type="sphere" size="0.5" mass="1" contype="0" conaffinity="0"/></body></worldbody></mujoco>""")
    d = mujoco.MjData(m); d.qvel[0] = v * math.cos(th); d.qvel[2] = v * math.sin(th); r = Rec(2)
    def cb(t):
        x = v * math.cos(th) * t; z = 0.3 + v * math.sin(th) * t - 0.5 * G * t * t
        e = math.hypot(d.xpos[1][0] - x, d.xpos[1][2] - z)
        r.add(t, [pose(d, 1), ([x, 0, z], [1, 0, 0, 0])], d.xpos[1][2], z, e)
    run(m, d, 2.34, cb)
    scene(key="projectile", title="포물선 운동", formula="x = v cosθ·t,  z = v sinθ·t − ½gt²", qty="높이 z (m)",
          bodies=[dict(shape="sphere", size=[0.5], color="sim", trail=True), dict(shape="sphere", size=[0.5], ghost=True)],
          ground=dict(size=22, center=[19, 0]), cam=dict(pos=[17, -24, 8], target=[19, 0, 3.2]), rec=r)


# 3. 대진폭 진자 -------------------------------------------------------------
def s_pendulum():
    L, th0 = 1.5, 2.5
    m = M(f"""<mujoco><option timestep="0.0005" integrator="RK4" gravity="0 0 -{G}"/>
      <worldbody><body pos="0 0 2"><joint type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -{L}" size="0.0001" mass="0" contype="0" conaffinity="0"/>
        <inertial pos="0 0 -{L}" mass="1" diaginertia="1e-9 1e-9 1e-9"/></body></worldbody></mujoco>""")
    d = mujoco.MjData(m); d.qpos[0] = th0; r = Rec(3)
    sol = solve_ivp(lambda t, y: [y[1], -G / L * math.sin(y[0])], [0, 12], [th0, 0],
                    rtol=1e-12, atol=1e-12, dense_output=True)
    w = math.sqrt(G / L)
    def cb(t):
        te = sol.sol(t)[0]; ts = th0 * math.cos(w * t)
        r.add(t, [pose(d, 1), ([0, 0, 2], qy(te)), ([0, 0, 2], qy(ts))], d.qpos[0], te, abs(d.qpos[0] - te))
    run(m, d, 12, cb)
    scene(key="pendulum", title="대진폭 진자 (θ₀ = 2.5 rad)", formula="θ̈ = −(g/L) sinθ   (정확해) vs  θ₀cos(√(g/L)t)  (소진폭 근사)",
          qty="각도 θ (rad)",
          bodies=[dict(shape="pend", size=[L, 0.15], color="sim"), dict(shape="pend", size=[L, 0.15], ghost=True),
                  dict(shape="pend", size=[L, 0.15], ghost=True, alt=True)],
          ground=dict(size=3), cam=dict(pos=[0.3, -6, 2], target=[0, 0, 1.6]), rec=r,
          legend=["MuJoCo", "정확해 (ODE)", "소진폭 근사 — 점점 어긋남"])


# 4. 스프링-감쇠 -------------------------------------------------------------
def s_spring():
    k, mass, x0 = 40.0, 1.0, 0.6
    wn = math.sqrt(k / mass); zetas = [0.0, 0.1, 0.5, 1.0, 2.0]
    bodies_xml = ""
    for i, z in enumerate(zetas):
        y = (i - 2) * 0.7
        bodies_xml += f"""<body pos="0 {y} 0.2"><joint type="slide" axis="1 0 0" stiffness="{k}" damping="{2*z*wn*mass}"/>
          <geom type="box" size="0.2 0.2 0.2" mass="{mass}" contype="0" conaffinity="0"/></body>"""
    m = M(f"""<mujoco><option timestep="0.0005" integrator="RK4" gravity="0 0 0"/><worldbody>{bodies_xml}</worldbody></mujoco>""")
    d = mujoco.MjData(m); d.qpos[:] = x0; r = Rec(10)

    def xref(z, t):
        if z < 1:
            wd = wn * math.sqrt(1 - z * z)
            return x0 * math.exp(-z * wn * t) * (math.cos(wd * t) + z * wn / wd * math.sin(wd * t))
        if z == 1:
            return x0 * (1 + wn * t) * math.exp(-wn * t)
        s = wn * math.sqrt(z * z - 1); r1, r2 = -z * wn + s, -z * wn - s
        return x0 * (r2 * math.exp(r1 * t) - r1 * math.exp(r2 * t)) / (r2 - r1)

    def cb(t):
        poses, errs = [], []
        for i, z in enumerate(zetas):
            poses.append(pose(d, i + 1))
        for i, z in enumerate(zetas):
            xr = xref(z, t); y = (i - 2) * 0.7
            poses.append(([xr, y, 0.2], [1, 0, 0, 0])); errs.append(abs(d.qpos[i] - xr))
        r.add(t, poses, d.qpos[1], xref(0.1, t), max(errs))
    run(m, d, 4, cb)
    b = [dict(shape="box", size=[0.2, 0.2, 0.2], color=c, spring=[-1.4, (i - 2) * 0.7, 0.2], label=f"ζ={z}")
         for i, (z, c) in enumerate(zip(zetas, ["c1", "c2", "c3", "c4", "c5"]))]
    b += [dict(shape="box", size=[0.2, 0.2, 0.2], ghost=True) for _ in zetas]
    scene(key="spring", title="스프링-감쇠 진동 (ζ = 0, 0.1, 0.5, 1, 2)", formula="mẍ + cẋ + kx = 0,  c = 2ζ√(km)",
          qty="x (m), ζ=0.1 상자", bodies=b, ground=dict(size=3), wall=[-1.4, 0, 0.3, 0.1, 1.9, 0.3],
          cam=dict(pos=[1.5, -4.2, 3.2], target=[-0.2, 0, 0]), rec=r)


# 5. 충돌과 운동량 -----------------------------------------------------------
def s_collision():
    m = M("""<mujoco><option timestep="0.0005" integrator="RK4" gravity="0 0 0"/><worldbody>
      <body pos="-1.5 0 1"><freejoint/><geom type="sphere" size="0.25" mass="1"/></body>
      <body pos="1.5 0.15 1"><freejoint/><geom type="sphere" size="0.36" mass="3"/></body></worldbody></mujoco>""")
    d = mujoco.MjData(m); d.qvel[0] = 2.0; d.qvel[6] = -0.5; d.qvel[3:6] = [0, 0, 0]
    ms = np.array([1.0, 3.0]); r = Rec(4)
    mujoco.mj_forward(m, d)
    c0 = (ms[0] * d.xpos[1] + ms[1] * d.xpos[2]) / ms.sum()
    vc = (ms[0] * d.qvel[0:3] + ms[1] * d.qvel[6:9]) / ms.sum()
    def cb(t):
        c = (ms[0] * d.xpos[1] + ms[1] * d.xpos[2]) / ms.sum(); cr = c0 + vc * t
        p = ms[0] * d.qvel[0] + ms[1] * d.qvel[6]
        r.add(t, [pose(d, 1), pose(d, 2), (c, [1, 0, 0, 0]), (cr, [1, 0, 0, 0])], p, ms.sum() * vc[0],
              float(np.linalg.norm(c - cr)))
    run(m, d, 3, cb)
    scene(key="collision", title="충돌과 운동량 보존", formula="Σmᵢvᵢ = const  →  질량중심 x_cm = x₀ + v_cm·t",
          qty="총 운동량 pₓ (kg·m/s)",
          bodies=[dict(shape="sphere", size=[0.25], color="c1", stripes=True), dict(shape="sphere", size=[0.36], color="c3", stripes=True),
                  dict(shape="sphere", size=[0.07], color="sim", trail=True), dict(shape="sphere", size=[0.1], ghost=True)],
          ground=dict(size=4), cam=dict(pos=[0, -6, 4], target=[0, 0, 1]), rec=r,
          legend=["가벼운 공 1 kg", "무거운 공 3 kg", "질량중심 (MuJoCo)", "질량중심 (해석해)"])


# 6. 자니베코프 효과 ---------------------------------------------------------
def s_dzhanibekov():
    I = np.array([1.0, 2.0, 3.0])
    m = M(f"""<mujoco><option timestep="0.0005" integrator="RK4" gravity="0 0 0"/><worldbody>
      <body pos="0 0 1.5"><freejoint/><inertial pos="0 0 0" mass="1" diaginertia="1 2 3"/></body></worldbody></mujoco>""")
    d = mujoco.MjData(m); d.qvel[3:6] = [0.01, 5.0, 0.01]; r = Rec(1)
    mujoco.mj_forward(m, d)
    def Lw():
        return d.xmat[1].reshape(3, 3) @ (I * d.qvel[3:6])
    L0 = Lw()
    def cb(t):
        Lv = Lw()
        r.add(t, [pose(d, 1)], np.linalg.norm(Lv), np.linalg.norm(L0), np.linalg.norm(Lv - L0) / np.linalg.norm(L0))
    run(m, d, 20, cb)
    scene(key="dzhanibekov", title="자니베코프 효과 (중간축 회전)", formula="L = R·I·ω = const,  I = (1, 2, 3)",
          qty="각운동량 크기 |L| (주황 화살표 = L 방향)",
          bodies=[dict(shape="tee", size=[0.7], color="sim")], ground=dict(size=3),
          cam=dict(pos=[3, -3.5, 2.8], target=[0, 0, 1.5]), rec=r, Larrow=rnd(L0 / np.linalg.norm(L0)))


# 7. 경사면 ------------------------------------------------------------------
def s_incline():
    th = math.radians(30); s, c = math.sin(th), math.cos(th)
    m = M(f"""<mujoco><option timestep="0.0005" gravity="{G*s} 0 {-G*c}" cone="elliptic" impratio="10"/><worldbody>
      <geom type="plane" size="4 2 0.1" friction="0 0 0"/>
      <body pos="-3 -0.8 0.15"><freejoint/><geom type="box" size="0.15 0.15 0.15" mass="1" friction="0.3 0 0"/></body>
      <body pos="-3 0 0.15"><freejoint/><geom type="box" size="0.15 0.15 0.15" mass="1" friction="0.7 0 0"/></body>
      <body pos="-3 0.8 0.15"><freejoint/><geom type="sphere" size="0.15" mass="1" friction="1 0.005 0.0001" condim="6"/></body>
      </worldbody></mujoco>""")
    d = mujoco.MjData(m); r = Rec(6)
    a1 = G * (s - 0.3 * c); a3 = 5 / 7 * G * s
    def cb(t):
        x1 = -3 + 0.5 * a1 * t * t; x3 = -3 + 0.5 * a3 * t * t
        poses = [pose(d, 1), pose(d, 2), pose(d, 3),
                 ([x1, -0.8, 0.15], [1, 0, 0, 0]), ([-3, 0, 0.15], [1, 0, 0, 0]), ([x3, 0.8, 0.15], qy((x3 + 3) / 0.15))]
        e = max(abs(d.xpos[1][0] - x1), abs(d.xpos[2][0] + 3), abs(d.xpos[3][0] - x3))
        r.add(t, poses, d.xpos[3][0] + 3, x3 + 3, e)
    run(m, d, 1.9, cb)
    scene(key="incline", title="경사면 30°: 미끄러짐 · 정지 · 구름", formula="a = g(sinθ − μcosθ)  |  μ > tanθ → 정지  |  구: a = (5/7)g sinθ",
          qty="구의 이동거리 (m)",
          bodies=[dict(shape="box", size=[0.15] * 3, color="c1", label="μ=0.3 미끄러짐"),
                  dict(shape="box", size=[0.15] * 3, color="c3", label="μ=0.7 정지"),
                  dict(shape="sphere", size=[0.15], color="c4", stripes=True, label="구 구름"),
                  dict(shape="box", size=[0.15] * 3, ghost=True), dict(shape="box", size=[0.15] * 3, ghost=True),
                  dict(shape="sphere", size=[0.15], ghost=True)],
          ground=dict(size=4, half=[4, 1.4], tilt=30), cam=dict(pos=[-0.5, -7.5, 2.0], target=[-0.6, 0, -0.6]), rec=r)


# 8. 자이로스코프 ------------------------------------------------------------
def s_gyro():
    rr, mass, R = 0.5, 1.0, 0.3
    I3, I1 = 0.5 * mass * R * R, 0.25 * mass * R * R
    w = 120.0; Om = mass * G * rr / (I3 * w)
    m = M(f"""<mujoco><option timestep="0.0001" integrator="RK4" gravity="0 0 -{G}"/><worldbody>
      <body pos="0 0 1.2"><joint type="ball"/><inertial pos="{rr} 0 0" mass="{mass}" diaginertia="{I3} {I1} {I1}"/></body>
      </worldbody></mujoco>""")
    d = mujoco.MjData(m); d.qvel[0] = w; d.qvel[2] = Om; r = Rec(2)
    phis = [0.0]
    def cb(t):
        ax = d.xmat[1].reshape(3, 3)[:, 0]
        ph = math.atan2(ax[1], ax[0])
        while ph < phis[-1] - math.pi: ph += 2 * math.pi
        phis.append(ph)
        qg = qmul([math.cos(Om * t / 2), 0, 0, math.sin(Om * t / 2)], [math.cos(w * t / 2), math.sin(w * t / 2), 0, 0])
        r.add(t, [pose(d, 1), ([0, 0, 1.2], qg)], ph, Om * t, abs(ph - Om * t))
    run(m, d, 2 * math.pi / Om + 0.2, cb)
    scene(key="gyro", title="자이로스코프 세차", formula=f"Ω = mgr / (I₃ω) = {Om:.4f} rad/s   (스핀 ω = {w:.0f} rad/s)",
          qty="세차 각 φ (rad)",
          bodies=[dict(shape="gyro", size=[rr, R], color="sim"), dict(shape="gyro", size=[rr, R], ghost=True)],
          ground=dict(size=2.5), post=[0, 0, 1.2], cam=dict(pos=[2.2, -2.6, 2.6], target=[0, 0, 1.0]), rec=r)


for f in [s_pendulum, s_freefall, s_projectile, s_spring, s_collision, s_dzhanibekov, s_incline, s_gyro]:
    f()

out = []
for s in SCENES:
    r = s.pop("rec")
    s.update(t=r.t, tr=r.tr, sim=rnd(r.sim, 6), ref=rnd(r.ref, 6), err=[float(f"{e:.3g}") for e in r.err],
             maxErr=float(max(r.err)))
    out.append(s)
    print(f"{s['key']:<12} frames {len(r.t):4d}  최대오차 {max(r.err):.2e}")
json.dump(dict(version=mujoco.__version__, scenes=out), open("scenes.json", "w"), separators=(",", ":"))
