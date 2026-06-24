# ⚡ Lightning AI에서 ACE-Step 1.5 XL + LoRA 학습하기(L4 24GB, 무료)

이 가이드는 **Lightning AI Studios**의 **무료 NVIDIA L4 GPU(24GB)**에서 **ACE-Step 1.5 XL(4B)**을 실행하고, 자신의 스타일로 **LoRA를 학습**하는 방법을 설명합니다.

> **왜 Snowflake가 아니라 Lightning AI인가?**
> - **Snowflake trial은 카드가 없으면 External Access가 완전히 막힙니다.** 패키지/모델 다운로드가 불가능합니다.
> - Lightning AI는 **L4 24GB VRAM**과 월 약 **17~22 무료 GPU 시간**을 제공합니다(15 credits = $15, L4는 약 0.7 credits/hour).
> - **영구 저장소**가 있어 XL 가중치와 학습된 LoRA가 세션 사이에 사라지지 않습니다(Kaggle/Snowflake와 다름).
> - 전체 인터넷, `sudo`, SSH가 되는 일반 Linux Studio라 Snowflake보다 훨씬 단순합니다.

> **⚠️ 아키텍처 참고:** LoRA 학습에는 React `ace-step-ui`가 아니라 **ACE-Step 네이티브 Gradio 앱**(`app.py`)을 실행합니다. **LoRA Training** 탭은 이 앱에만 있으며, 생성과 학습을 한 창에서 처리합니다.

---

## 1. 가입과 Studio 생성

1. https://lightning.ai 에 가입합니다(무료). 로그인 후 무료 CPU Studio 1개와 월 약 22 GPU 시간이 제공됩니다.
2. **+ New Studio** → PyTorch/CUDA 템플릿(예: GPU 이미지가 포함된 기본 “Code”) 선택 → **Start**.
3. 브라우저에서 VS Code / Jupyter와 터미널이 열립니다.

## 2. L4 GPU 켜기

1. Studio 오른쪽 위에서 compute 전환 메뉴를 찾습니다(처음에는 CPU).
2. **GPU → L4 (24 GB)** 선택 → Studio가 GPU로 재시작됩니다(파일은 보존).
3. 💡 **절약:** GPU 시간은 GPU가 켜져 있는 동안만 차감됩니다. 작업하지 않을 때는 CPU로 전환하거나 Studio를 중지하세요.

```bash
nvidia-smi          # NVIDIA L4, 약 24GB가 보여야 함
python --version    # ACE-Step은 Python >= 3.11 필요(끝의 참고 참고)
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 3. 설치(터미널)

```bash
# 0) 시스템 의존성(Lightning에서는 sudo 가능)
sudo apt-get update && sudo apt-get install -y ffmpeg git

# 1) 네이티브 ACE-Step 1.5 엔진 클론(~/는 영구 저장소)
cd ~
git clone https://github.com/ace-step/ACE-Step-1.5.git
cd ACE-Step-1.5

# 2) 패키지 자체만 설치, 의존성은 제외(사전 설치된 CUDA torch를 망가뜨리지 않음)
pip install -e . --no-deps

# 3) 런타임 의존성(torch 제외)
pip install --no-cache-dir \
  "transformers>=4.51.0,<4.58.0" "diffusers>=0.37.0" "accelerate>=1.12.0" \
  "huggingface_hub[hf_xet]>=0.34.0,<1.0" \
  "soundfile>=0.13.1" librosa soxr loguru einops scipy diskcache numba \
  "vector-quantize-pytorch>=1.27.15" pytorch-wavelets pywavelets toml modelscope matplotlib \
  gradio nano-vllm

python -c "import torch,transformers; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), '| transformers', transformers.__version__)"
```

> `nano-vllm` 설치가 실패하면 건너뛰어도 됩니다. LoRA 학습에는 LM이 필요하지 않습니다.

---

## 4. Gradio 실행 + 공개 URL

```bash
cd ~/ACE-Step-1.5
export HF_HOME=~/hf            # 영구 저장소에 모델 캐시(XL은 한 번만 다운로드)

# Gradio 실행: --config acestep-v15-xl-turbo = 4B XL turbo
python app.py --config acestep-v15-xl-turbo --port 7860
```

브라우저에서 UI를 여는 방법은 두 가지입니다.

**A) Lightning 내장 포트 포워딩(가장 쉬움):** Studio에서 **Ports** 플러그인을 열고 포트 **7860**을 추가한 뒤 **Open**을 누르면 공개 URL이 생성됩니다.

**B) cloudflared(범용 터널)** — 새 터미널에서 실행합니다. 첫 번째 터미널의 Gradio는 계속 켜 둡니다.

```bash
wget -q -O ~/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x ~/cloudflared
~/cloudflared tunnel --url http://localhost:7860 --no-autoupdate
# 출력된 https://....trycloudflare.com 링크 복사
```

---

## 5. LoRA 학습 — 단계별

UI 열기 → **Initialize Service** → 아래쪽 **LoRA Training** 탭.

### 1단계. 데이터셋(자신의 오리지널 트랙)

파일을 예를 들어 `~/dataset/` 폴더에 넣습니다.

```
dataset/
├── song1.mp3            # 오디오(.mp3/.wav/.flac/.ogg/.opus)
├── song1.lyrics.txt     # 가사(또는 song1.txt)
├── song1.json           # 메타데이터(선택)
└── ...
```

`song1.json`의 모든 필드는 선택입니다.

```json
{
  "caption": "A high-energy J-pop track with synthesizer leads and fast tempo",
  "bpm": 190,
  "keyscale": "D major",
  "timesignature": "4",
  "language": "ru"
}
```

- **개수:** 스타일 학습에는 8~20개 트랙을 권장합니다(양보다 품질).
- **BPM/Key:** https://vocalremover.org/key-bpm-finder 에서 얻고 Export CSV로 폴더에 넣는 것이 쉽습니다.
- **Caption:** 직접 작성하거나 UI의 **Auto Label**(LM `acestep-5Hz-lm`)로 생성할 수 있습니다.

### 2단계. Scan과 라벨 확인

**LoRA Training** 탭 → 폴더 경로 입력 → **Scan**. 각 트랙에 가사와 caption이 있는지 확인합니다(Labeled = ✅). 필요하면 **Auto Label**을 실행하고 기록을 수정한 뒤 수정할 때마다 **Save** → **Save Dataset**으로 JSON을 내보냅니다.

### 3단계. 텐서 전처리

텐서 저장 경로를 지정하고 실행합니다. 💡 caption 생성을 위해 LM을 사용했다면, VRAM 확보를 위해 LM 없이 Gradio를 재시작하세요.

### 4단계. 학습(LoKr 권장)

**Train LoKr** 탭(LoRA보다 약 10배 빠름, 한 시간 대신 몇 분) → 텐서 경로 → 로드.

- **Max Epochs:** 약 100개 트랙 → 500, 10~20개 트랙 → 약 800.
- **Batch Size:** 1(24GB에서는 2도 시도 가능).
- **gradient_checkpointing:** OOM이면 켭니다.
- **Start Training** → loss를 확인합니다.

### 5단계. 사용

Gradio를 LM 없이 재시작 → 학습된 LoRA/LoKr 파일 로드 → 생성 🎶

### 대안: localhost:8001 REST API

```bash
curl -X POST http://localhost:8001/v1/training/start_lokr \
  -H 'Content-Type: application/json' \
  -d '{"tensor_dir":"~/tensors","output_dir":"~/lokr_output","lokr_linear_dim":64,"lokr_linear_alpha":128,"lokr_factor":-1,"lokr_weight_decompose":true,"learning_rate":0.03,"train_epochs":500,"train_batch_size":1,"gradient_accumulation":4,"save_every_n_epochs":5}'
```

자세한 튜토리얼: https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/LoRA_Training_Tutorial.md

---

## 6. 저장과 GPU 시간 절약

- **영구 저장소:** `~/`의 모든 파일(XL 가중치 `~/hf`, LoRA `~/lokr_output`)은 재시작 및 GPU→CPU 전환 후에도 남습니다. Kaggle/Snowflake와 달리 수동으로 클라우드에 올릴 필요가 없습니다.
- **GPU 시간 절약:** 작업 후 compute를 **CPU**로 바꾸거나 Studio를 중지하세요. GPU 시간이 차감되는 것은 GPU가 활성화되어 있을 때뿐입니다. LoKr 학습은 빠르므로 월 17~22시간이면 충분합니다.

---

## 7. 참고

- **Python < 3.11:** ACE-Step은 Python 3.11 이상이 필요합니다. 이미지가 3.10이면 `conda create -y -n ace python=3.11 && conda activate ace`로 환경을 만들고 CUDA용 torch를 다시 설치한 뒤(https://pytorch.org/get-started/locally/ ) 섹션 3을 반복하세요.
- **학습 중 OOM:** `gradient_checkpointing`을 켜고, 트랙 길이를 줄이고, LoKr을 사용하세요.
- **전처리 중 OOM:** 전처리 전에 LM 없이 Gradio를 재시작하세요.
- **저작권:** 자신의 오리지널 작품으로만 LoRA를 학습하세요.

---

## TL;DR

1. lightning.ai → New Studio → compute를 **L4 (24 GB)**로 전환.
2. 터미널에서 엔진 설치(섹션 3).
3. `python app.py --config acestep-v15-xl-turbo --port 7860` → 포트 7860 열기(Ports 또는 cloudflared).
4. **LoRA Training** 탭 → 데이터셋 → Scan → 전처리 → **Train LoKr** → LoRA 로드 → 생성.
5. 작업 후 CPU로 전환하거나 Studio 중지(파일은 보존).
