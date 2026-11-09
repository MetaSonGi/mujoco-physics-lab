"""MuJoCo 물리 엔진 수학적 검증 스위트.

각 테스트는 MuJoCo 시뮬레이션 결과를 해석해(닫힌 형태의 해)와 비교하고
허용 오차 안에 드는지 판정한다. 결과는 results.json 으로 저장되어 웹 페이지가 읽는다.

실행:  python mujoco_physics_tests.py
"""
import json
import math
import numpy as np
import mujoco
from scipy.special import ellipk

G = 9.81
RESULTS = []


def model(xml):
    return mujoco.MjModel.from_xml_string(xml)


def ds(arr, n=300):
    """그래프용 다운샘플."""
    arr = np.asarray(arr)
    if len(arr) <= n:
        return arr.tolist()
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return arr[idx].tolist()


def record(key, title, formula, measured, expected, tol, unit="", note="", series=None, extra=None):
    rel = abs(measured - expected) / abs(expected) if expected != 0 else abs(measured)
    passed = bool(rel <= tol)
    RESULTS.append(dict(key=key, title=title, formula=formula, measured=float(measured),
                        expected=float(expected), rel_error=float(rel), tol=tol, unit=unit,
                        passed=passed, note=note, series=series or {}, extra=extra or {}))
    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] {title:<34} 측정 {measured:.6g}{unit}  해석 {expected:.6g}{unit}  상대오차 {rel:.2e} (허용 {tol:.0e})")


def zero_cross_period(t, x):
    """상향 영점 통과 사이 평균 간격 (선형보간)."""
    ups = []
    for i in range(1, len(x)):
        if x[i - 1] < 0 <= x[i]:
            ups.append(t[i - 1] + (0 - x[i - 1]) * (t[i] - t[i - 1]) / (x[i] - x[i - 1]))
    return float(np.mean(np.diff(ups))), ups


# ---------------------------------------------------------------- 1. 자유낙하
def test_free_fall():
    m = model(f"""<mujoco><option timestep="0.001" integrator="RK4" gravity="0 0 -{G}"/>
      <worldbody><body pos="0 0 10"><freejoint/>
        <geom type="sphere" size="0.1" mass="1" contype="0" conaffinity="0"/></body></worldbody></mujoco>""")
    d = mujoco.MjData(m)
    T = 1.0
    ts, zs, za = [], [], []
    while d.time < T - 1e-12:
        mujoco.mj_step(m, d)
        ts.append(d.time); zs.append(d.qpos[2]); za.append(10 - 0.5 * G * d.time ** 2)
    err = np.max(np.abs(np.array(zs) - np.array(za)))
    record("free_fall", "자유낙하 z(t)", "z = z₀ − ½gt²", zs[-1], za[-1], 1e-9, " m",
           note=f"전 구간 최대 절대오차 {err:.2e} m (RK4는 2차 다항식을 정확히 적분)",
           series=dict(t=ds(ts), sim=ds(zs), ref=ds(za)))


# ---------------------------------------------------------------- 2. 포물선 운동
def test_projectile():
    v, th = 20.0, math.radians(35)
    m = model(f"""<mujoco><option timestep="0.0005" integrator="RK4" gravity="0 0 -{G}"/>
      <worldbody><body pos="0 0 0"><freejoint/>
        <geom type="sphere" size="0.05" mass="1" contype="0" conaffinity="0"/></body></worldbody></mujoco>""")
    d = mujoco.MjData(m)
    d.qvel[0] = v * math.cos(th); d.qvel[2] = v * math.sin(th)
    xs, zs, xr, zr = [], [], [], []
    prev = None
    while True:
        p = (d.qpos[0], d.qpos[2])
        mujoco.mj_step(m, d)
        xs.append(d.qpos[0]); zs.append(d.qpos[2])
        t = d.time
        xr.append(v * math.cos(th) * t); zr.append(v * math.sin(th) * t - 0.5 * G * t * t)
        if d.qpos[2] < 0 and t > 0.1:
            x0, z0 = p; x1, z1 = d.qpos[0], d.qpos[2]
            R = x0 + (0 - z0) * (x1 - x0) / (z1 - z0)
            break
    R_ref = v ** 2 * math.sin(2 * th) / G
    H = max(zs); H_ref = (v * math.sin(th)) ** 2 / (2 * G)
    record("projectile", "포물선 사거리 R", "R = v²·sin2θ / g", R, R_ref, 1e-5, " m",
           note=f"최고점 H: 측정 {H:.5f} m / 해석 {H_ref:.5f} m",
           series=dict(x=ds(xs), sim=ds(zs), ref=ds(zr), xref=ds(xr)))


# ---------------------------------------------------------------- 3. 단진자
def pendulum_xml(L, dt, integ="RK4"):
    return f"""<mujoco><option timestep="{dt}" integrator="{integ}" gravity="0 0 -{G}">
      <flag energy="enable"/></option>
      <worldbody><body pos="0 0 0"><joint name="h" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 -{L}" mass="1" diaginertia="1e-9 1e-9 1e-9"/></body></worldbody></mujoco>"""


def run_pendulum(L, th0, T, dt=0.0005, integ="RK4"):
    m = model(pendulum_xml(L, dt, integ)); d = mujoco.MjData(m)
    d.qpos[0] = th0
    ts, qs, es = [], [], []
    mujoco.mj_forward(m, d)
    while d.time < T - 1e-12:
        mujoco.mj_step(m, d)
        ts.append(d.time); qs.append(d.qpos[0]); es.append(d.energy[0] + d.energy[1])
    return np.array(ts), np.array(qs), np.array(es)


def test_pendulum():
    L = 1.0
    T0 = 2 * math.pi * math.sqrt(L / G)
    t, q, _ = run_pendulum(L, 0.02, 10 * T0)
    P, _ = zero_cross_period(t, q)
    ref = 0.02 * np.cos(2 * math.pi * t / T0)
    record("pend_small", "단진자 주기 (θ₀=0.02 rad)", "T = 2π√(L/g)", P, T0, 1e-4, " s",
           note="소진폭 근사. 차이는 sinθ≈θ 근사 오차(≈θ₀²/16) 수준",
           series=dict(t=ds(t), sim=ds(q), ref=ds(ref)))
    # 대진폭: 타원적분 정확해
    curve = []
    for th0 in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        Te = 4 * math.sqrt(L / G) * ellipk(math.sin(th0 / 2) ** 2)
        t, q, _ = run_pendulum(L, th0, 3 * Te + 0.5)
        P, _ = zero_cross_period(t, q)
        curve.append(dict(th0=th0, sim=P, exact=Te, small=T0))
    th0 = 2.5
    c = [x for x in curve if x["th0"] == th0][0]
    record("pend_large", "단진자 주기 (θ₀=2.5 rad)", "T = 4√(L/g)·K(sin²(θ₀/2))", c["sim"], c["exact"], 1e-4, " s",
           note=f"정확해는 소진폭 공식({T0:.4f}s)보다 {100*(c['exact']/T0-1):.1f}% 길다. MuJoCo는 정확해를 따라감",
           series=dict(curve=curve))


# ---------------------------------------------------------------- 4. 에너지 보존 / 적분기 비교
def test_energy():
    out = {}
    T = 30.0
    for integ in ["Euler", "implicitfast", "RK4"]:
        t, q, e = run_pendulum(1.0, 1.0, T, dt=0.01, integ=integ)
        e0 = -G * math.cos(1.0)  # 초기 위치에너지 (pivot 기준, m=1, L=1)
        drift = (e - e0) / abs(e0)
        out[integ] = dict(t=ds(t), drift=ds(drift), max=float(np.max(np.abs(drift))), final=float(drift[-1]))
    record("energy_rk4", "에너지 보존 (RK4, dt=10ms, 30s)", "E = ½mL²θ̇² − mgLcosθ = const",
           out["RK4"]["max"], 0.0, 1e-4, "", note="상대 에너지 편차의 최댓값 (기댓값 0)",
           series=out)
    RESULTS[-1]["rel_error"] = float(out["RK4"]["max"]); RESULTS[-1]["passed"] = bool(out["RK4"]["max"] < 1e-4)
    for k in ["Euler", "implicitfast"]:
        print(f"       {k:<13} 최대 에너지 편차 {out[k]['max']:.2e}  최종 {out[k]['final']:+.2e}")


# ---------------------------------------------------------------- 5. 적분기 수렴 차수
def test_convergence():
    th0, T = 1.0, 2.0
    _, qref, _ = run_pendulum(1.0, th0, T, dt=1e-5, integ="RK4")
    ref = qref[-1]
    dts = [0.02, 0.01, 0.005, 0.0025, 0.00125]
    res = {}
    for integ in ["Euler", "RK4"]:
        errs = []
        for dt in dts:
            _, q, _ = run_pendulum(1.0, th0, T, dt=dt, integ=integ)
            errs.append(abs(q[-1] - ref))
        slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
        res[integ] = dict(dts=dts, errs=errs, order=float(slope))
    record("conv_euler", "수렴 차수 (Euler, 반암시적)", "오차 ∝ Δt¹", res["Euler"]["order"], 1.0, 0.1, "",
           note="log-log 기울기", series=res)
    record("conv_rk4", "수렴 차수 (RK4)", "오차 ∝ Δt⁴", res["RK4"]["order"], 4.0, 0.1, "",
           note="log-log 기울기", series=res)


# ---------------------------------------------------------------- 6. 스프링-감쇠 진동
def test_spring():
    k, mass = 40.0, 1.0
    wn = math.sqrt(k / mass)
    cases = {}
    worst = 0
    for zeta in [0.0, 0.1, 0.5, 1.0, 2.0]:
        c = 2 * zeta * wn * mass
        m = model(f"""<mujoco><option timestep="0.0005" integrator="RK4" gravity="0 0 0"/>
          <worldbody><body><joint type="slide" axis="1 0 0" stiffness="{k}" damping="{c}"/>
            <inertial pos="0 0 0" mass="{mass}" diaginertia="1 1 1"/></body></worldbody></mujoco>""")
        d = mujoco.MjData(m); x0 = 0.1; d.qpos[0] = x0
        ts, xs = [], []
        while d.time < 3.0 - 1e-12:
            mujoco.mj_step(m, d); ts.append(d.time); xs.append(d.qpos[0])
        t = np.array(ts)
        if zeta < 1:
            wd = wn * math.sqrt(1 - zeta ** 2)
            ref = x0 * np.exp(-zeta * wn * t) * (np.cos(wd * t) + zeta * wn / wd * np.sin(wd * t))
        elif zeta == 1:
            ref = x0 * (1 + wn * t) * np.exp(-wn * t)
        else:
            s = wn * math.sqrt(zeta ** 2 - 1)
            r1, r2 = -zeta * wn + s, -zeta * wn - s
            A = x0 * r2 / (r2 - r1); B = -x0 * r1 / (r2 - r1)
            ref = A * np.exp(r1 * t) + B * np.exp(r2 * t)
        e = float(np.max(np.abs(np.array(xs) - ref)) / x0)
        worst = max(worst, e)
        cases[str(zeta)] = dict(t=ds(t), sim=ds(xs), ref=ds(ref), err=e)
    record("spring", "감쇠 진동 x(t) (ζ=0~2)", "mẍ + cẋ + kx = 0", worst, 0.0, 1e-6, "",
           note="5개 감쇠비(무감쇠·부족·임계·과감쇠) 중 최대 정규화 오차", series=cases)
    RESULTS[-1]["rel_error"] = float(worst); RESULTS[-1]["passed"] = bool(worst < 1e-6)


# ---------------------------------------------------------------- 7. 운동량 보존 (충돌)
def test_momentum():
    m = model("""<mujoco><option timestep="0.0005" integrator="RK4" gravity="0 0 0"/>
      <worldbody>
        <body name="a" pos="-1 0 0"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
        <body name="b" pos="1 0.05 0"><freejoint/><geom type="sphere" size="0.1" mass="3"/></body>
      </worldbody></mujoco>""")
    d = mujoco.MjData(m)
    d.qvel[0] = 2.0; d.qvel[6] = -0.5
    ma, mb = 1.0, 3.0
    p0 = ma * d.qvel[0:3] + mb * d.qvel[6:9]
    ts, px, py, va, vb, ke = [], [], [], [], [], []
    while d.time < 1.5:
        mujoco.mj_step(m, d)
        p = ma * d.qvel[0:3] + mb * d.qvel[6:9]
        ts.append(d.time); px.append(p[0]); py.append(p[1]); va.append(d.qvel[0]); vb.append(d.qvel[6])
        ke.append(0.5 * ma * np.dot(d.qvel[0:3], d.qvel[0:3]) + 0.5 * mb * np.dot(d.qvel[6:9], d.qvel[6:9]))
    p1 = ma * d.qvel[0:3] + mb * d.qvel[6:9]
    dp = float(np.linalg.norm(p1 - p0))
    record("momentum", "선운동량 보존 (비스듬한 충돌)", "Σmᵢvᵢ = const", float(p1[0]), float(p0[0]), 1e-6, " kg·m/s",
           note=f"|Δp| = {dp:.2e}. 운동에너지 {ke[0]:.3f}→{ke[-1]:.3f} J (기본 solref는 임계감쇠 연성 접촉 → 비탄성 충돌, 운동량은 그대로 보존)",
           series=dict(t=ds(ts), px=ds(px), py=ds(py), va=ds(va), vb=ds(vb), ke=ds(ke)))


# ---------------------------------------------------------------- 8. 각운동량 보존 + 중간축 정리
def test_angular():
    I = np.array([1.0, 2.0, 3.0])
    m = model(f"""<mujoco><option timestep="0.0002" integrator="RK4" gravity="0 0 0"/>
      <worldbody><body><freejoint/>
        <inertial pos="0 0 0" mass="1" diaginertia="{I[0]} {I[1]} {I[2]}"/></body></worldbody></mujoco>""")
    d = mujoco.MjData(m)
    d.qvel[3:6] = [0.01, 5.0, 0.01]   # 중간축(y) 근처 회전 → 불안정 (자니베코프 효과)

    def Lworld():
        R = d.xmat[1].reshape(3, 3)
        w_body = d.qvel[3:6]           # freejoint 각속도는 body 좌표계
        return R @ (I * w_body)
    mujoco.mj_forward(m, d)
    L0 = Lworld()
    ts, wx, wy, wz, Ln = [], [], [], [], []
    E0 = 0.5 * np.sum(I * d.qvel[3:6] ** 2)
    while d.time < 20:
        mujoco.mj_step(m, d)
        Lw = Lworld()
        ts.append(d.time); wx.append(d.qvel[3]); wy.append(d.qvel[4]); wz.append(d.qvel[5])
        Ln.append(np.linalg.norm(Lw - L0) / np.linalg.norm(L0))
    E1 = 0.5 * np.sum(I * d.qvel[3:6] ** 2)
    flips = int(np.sum(np.diff(np.sign(wy)) != 0))
    record("angular", "각운동량 보존 (비대칭 강체)", "L = R·I·ω = const (토크 0)", max(Ln), 0.0, 1e-6, "",
           note=f"중간축 회전이 {flips}회 뒤집힘(자니베코프 효과)에도 L은 보존. 회전에너지 상대변화 {abs(E1-E0)/E0:.1e}",
           series=dict(t=ds(ts, 600), wx=ds(wx, 600), wy=ds(wy, 600), wz=ds(wz, 600), Ln=ds(Ln, 600)))
    RESULTS[-1]["rel_error"] = float(max(Ln)); RESULTS[-1]["passed"] = bool(max(Ln) < 1e-6)


# ---------------------------------------------------------------- 9. 경사면 마찰 미끄러짐
def incline(geom, mu, ang_deg, T=1.5, cone="elliptic"):
    # 경사면 대신 중력 벡터를 θ만큼 기울임 (물리적으로 동치)
    a_ = math.radians(ang_deg)
    gx, gz = G * math.sin(a_), -G * math.cos(a_)
    m = model(f"""<mujoco><option timestep="0.0005" gravity="{gx} 0 {gz}" cone="{cone}" impratio="10"/>
      <worldbody><geom type="plane" size="50 5 0.1" friction="{mu} 0 0"/>
        <body pos="0 0 0.1"><freejoint/>{geom.format(mu=mu)}</body>
      </worldbody></mujoco>""")
    d = mujoco.MjData(m)
    ts, vs = [], []
    while d.time < T:
        mujoco.mj_step(m, d)
        ts.append(d.time); vs.append(np.linalg.norm(d.qvel[0:3]))
    t = np.array(ts); v = np.array(vs)
    sel = t > 0.3
    a = np.polyfit(t[sel], v[sel], 1)[0]
    return a, t, v


def test_incline():
    th = 30
    s = math.sin(math.radians(th)); c = math.cos(math.radians(th))
    box = '<geom type="box" size="0.1 0.1 0.1" mass="1" friction="{mu} 0 0"/>'
    a, t, v = incline(box, 0.3, th)
    ref = G * (s - 0.3 * c)
    record("incline_slide", "경사면 미끄러짐 (μ=0.3, 30°)", "a = g(sinθ − μcosθ)", a, ref, 0.01, " m/s²",
           note="쿨롱 마찰 가속도", series=dict(t=ds(t), sim=ds(v), ref=ds(ref * t)))
    a2, t2, v2 = incline(box, 0.7, th)
    record("incline_stick", "정지마찰 (μ=0.7 > tan30°)", "a = 0 (μ > tanθ)", v2[-1], 0.0, 1e-3, " m/s",
           note=f"1.5초 후 속도 {v2[-1]:.2e} m/s — 정지 유지", series=dict(t=ds(t2), sim=ds(v2)))
    RESULTS[-1]["rel_error"] = float(abs(v2[-1])); RESULTS[-1]["passed"] = bool(abs(v2[-1]) < 1e-3)
    ball = '<geom type="sphere" size="0.1" mass="1" friction="{mu} 0 0" condim="6"/>'
    a3, t3, v3 = incline(ball, 1.0, th)
    ref3 = 5 / 7 * G * s
    record("incline_roll", "구의 구름 운동 (미끄럼 없음)", "a = (5/7)g·sinθ", a3, ref3, 0.01, " m/s²",
           note=f"마찰 없는 미끄러짐이면 g·sinθ = {G*s:.3f}. 회전 관성 때문에 5/7배",
           series=dict(t=ds(t3), sim=ds(v3), ref=ds(ref3 * t3)))


# ---------------------------------------------------------------- 10. 자이로스코프 세차
def test_gyro():
    r, mass, R = 0.3, 1.0, 0.2
    I3 = 0.5 * mass * R ** 2
    I1 = 0.25 * mass * R ** 2
    w = 200.0
    m = model(f"""<mujoco><option timestep="0.0001" integrator="RK4" gravity="0 0 -{G}"/>
      <worldbody><body pos="0 0 1"><joint type="ball"/>
        <inertial pos="{r} 0 0" mass="{mass}" diaginertia="{I3} {I1} {I1}"/></body></worldbody></mujoco>""")
    d = mujoco.MjData(m)
    d.qvel[0] = w   # 몸체 x축(= 원판 대칭축) 둘레 스핀
    Om = mass * G * r / (I3 * w)
    # 순수 세차 초기조건: 세차 각속도를 미리 부여해 장동 최소화
    d.qvel[2] = Om
    ts, ph, zs = [], [], []
    while d.time < 4.0:
        mujoco.mj_step(m, d)
        ax = d.xmat[1].reshape(3, 3)[:, 0]
        ts.append(d.time); ph.append(math.atan2(ax[1], ax[0])); zs.append(ax[2])
    ph = np.unwrap(ph)
    rate = np.polyfit(ts, ph, 1)[0]
    record("gyro", "자이로 세차 각속도", "Ω = mgr / (I₃ω)", rate, Om, 0.01, " rad/s",
           note=f"스핀 ω={w:.0f} rad/s. 축 높이 변화(장동) 최대 {max(abs(np.array(zs))):.1e}",
           series=dict(t=ds(ts), sim=ds(ph), ref=ds(Om * np.array(ts))))


if __name__ == "__main__":
    print(f"MuJoCo {mujoco.__version__}\n")
    for f in [test_free_fall, test_projectile, test_pendulum, test_energy, test_convergence,
              test_spring, test_momentum, test_angular, test_incline, test_gyro]:
        f()
    n = sum(r["passed"] for r in RESULTS)
    print(f"\n{n}/{len(RESULTS)} 통과")
    with open("results.json", "w") as fp:
        json.dump(dict(version=mujoco.__version__, results=RESULTS), fp,
                  default=float, separators=(",", ":"))
    # 그래프용 경량본 (유효숫자 6자리)
    import re
    txt = open("results.json").read()
    txt = re.sub(r"-?\d+\.\d{7,}(e-?\d+)?", lambda m: f"{float(m.group(0)):.6g}", txt)
    open("results.json", "w").write(txt)
