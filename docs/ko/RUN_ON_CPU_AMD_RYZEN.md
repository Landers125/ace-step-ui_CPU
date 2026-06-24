# CPU에서 ACE-Step UI 실행하기(AMD Ryzen 5 5600G, 외장 GPU 없음)

이 문서는 외장 그래픽카드가 없는 Windows PC, 예를 들어 Radeon 내장 그래픽이 있는 **AMD Ryzen 5 5600G**와 16GB RAM 환경에서 `ace-step-ui`를 실행하는 방법을 설명합니다.

## 하드웨어 관련 핵심 내용

`ace-step-ui` 자체는 React 인터페이스와 Node/Express 서버일 뿐이므로 GPU가 필요하지 않습니다. 실제 음악 생성은 UI가 API로 연결하는 별도 엔진 [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)가 담당합니다.

내장 Radeon(Vega, gfx90c)은 생성 가속에 **사용할 수 없습니다**. PyTorch의 AMD 지원은 ROCm을 통해 동작하지만, ROCm은 APU 내장 그래픽을 지원하지 않습니다. 따라서 이 PC에서의 실질적인 실행 방식은 **CPU 모드**입니다. 느리지만 동작합니다.

## 필요한 것

- Node.js 18+ — https://nodejs.org/
- Python 3.11 — https://www.python.org/downloads/
- uv: `pip install uv`
- FFmpeg — https://ffmpeg.org/ (없으면 트랙 길이가 0:00으로 표시될 수 있음)
- Git — https://git-scm.com/

## 1단계. ACE-Step 1.5 엔진을 CPU 모드로 설치

기본 설치는 CUDA용 PyTorch 빌드를 설치하므로 이 PC에서는 동작하지 않습니다. **CPU 빌드**가 필요합니다.

```bash
git clone https://github.com/ace-step/ACE-Step-1.5
cd ACE-Step-1.5

uv venv
uv pip install -e .

# CPU 버전 PyTorch를 강제로 덮어 설치
uv pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cpu

cd ..
```

확인합니다. CPU 환경에서는 `False`가 정상입니다.

```bash
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 2단계. ACE-Step UI

```bash
git clone https://github.com/Landers125/ace-step-ui_CPU
cd ace-step-ui_CPU
setup.bat
```

스크립트가 엔진을 자동으로 찾을 수 있도록 두 폴더를 나란히 배치하세요.

```
any-folder
  ACE-Step-1.5
  ace-step-ui_CPU
```

엔진이 다른 위치에 있다면 다음처럼 지정합니다: `set ACESTEP_PATH=C:\path\to\ACE-Step-1.5`

## 3단계. 실행

```bash
cd ace-step-ui_CPU
start-all-cpu.bat
```

`start-all-cpu.bat` 스크립트는 GPU를 강제로 숨기고, 가벼운 DiT-only 모드(무거운 LLM 제외)를 켠 뒤 API + 백엔드 + 프론트엔드를 실행합니다. 브라우저에서 http://localhost:3000 이 열립니다.

스크립트가 설정하는 환경 변수:

| 변수 | 값 | 목적 |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `-1` | NVIDIA GPU 숨김 |
| `HIP_VISIBLE_DEVICES` | `-1` | AMD GPU 숨김 |
| `ACESTEP_LM_BACKEND` | `pt` | LLM의 PyTorch 백엔드 |
| `ACESTEP_INIT_LLM` | `false` | DiT-only, RAM 절약 |
| `ACESTEP_VAE_ON_CPU` | `1` | VAE를 CPU에서 실행 |

## 속도와 메모리를 위한 설정

- **Thinking Mode** — 끄기(DiT-only에서는 사용할 수 없음).
- **AI Enhance** — 끄기(LLM 필요).
- **Inference Steps** — 20부터 시작.
- **Batch Size** — 1.
- **Audio Duration** — 30~60초부터 시작.

> AI Enhance / Thinking을 다시 쓰고 싶고 RAM 여유가 있다면 `start-all-cpu.bat`에서 `ACESTEP_INIT_LLM=false`를 `true`로 바꾸세요. CPU에 LLM 0.6B가 로드되며 더 느려집니다.

## 문제 해결

| 증상 | 해결 |
| --- | --- |
| `ACE-Step not reachable` | API 창이 포트 8001에서 리슨 중인지 확인 |
| 길이가 0:00으로 표시됨 | FFmpeg를 설치하고 PATH에 추가 |
| 메모리 부족 | `ACESTEP_INIT_LLM=false`, Batch Size = 1, 짧은 길이 사용 |
| 생성이 너무 오래 걸림 | CPU에서는 정상입니다. Inference Steps와 길이를 줄이세요 |
