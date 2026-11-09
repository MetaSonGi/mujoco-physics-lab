# MuJoCo 물리 실험실

브라우저에서 MuJoCo 4.0(WebAssembly)을 직접 돌려 14가지 고전역학 실험을 해석해와 실시간으로 비교하는 페이지입니다.

**실행:** https://metasongi.github.io/mujoco-physics-lab/

## 실험
투사체와 공기 저항 · 단진자(큰 진폭) · 이중 진자(혼돈) · 강제 진동과 공진 · 충돌과 운동량 · 경사면 미끄러짐과 구름 · 애트우드 기계 · 강체 회전(자니베코프 효과) · 자이로스코프 세차 · 원뿔 진자 · 막대 진자 · 볼링공 미끄럼→구름 · 결합 진동자와 맥놀이 · 현수선

각 실험은 슬라이더로 변수를 바꾸면 모델을 다시 컴파일하고, 주황 점선 고스트로 같은 조건의 해석해를 그리며, 오차와 허용치로 PASS/FAIL을 판정합니다. 물체를 마우스로 잡아 끌 수 있고, 적분기(RK4 / implicitfast / Euler)를 바꿔 볼 수 있습니다.

## 파일
- `index.html` — 페이지 전체. three.js r128과 공식 [`@mujoco/mujoco`](https://www.npmjs.com/package/@mujoco/mujoco) 4.0.0 WASM 빌드(Apache-2.0, Google DeepMind)는 CDN(jsDelivr)에서 불러옵니다
- `python/mujoco_physics_tests.py` — Python `mujoco`로 14개 항목을 해석해와 비교하는 검증 스크립트
- `python/record3d.py` — 3D 재생용 궤적 기록 스크립트

로컬 실행: `python3 -m http.server` 후 http://localhost:8000 (파일을 직접 여는 `file://`로는 WASM이 로드되지 않습니다).
