# 🎵 Snowflake에서 ACE-Step 1.5 XL + LoRA 학습하기(GPU A10G 24GB)

이 가이드는 **Snowflake** 무료/트라이얼 리소스의 **NVIDIA A10G 24GB** GPU에서 **ACE-Step 1.5 XL(4B)**을 실행하고, 자신의 스타일로 **LoRA를 학습**하는 방법을 설명합니다.

> **왜 Kaggle P100이 아니라 Snowflake A10G인가?**
> - **24GB VRAM**(P100의 16GB 대비)이라 XL 생성과 LoRA 학습을 더 여유 있게 처리합니다.
> - LoRA 학습에는 **최소 16GB, 권장 20GB+**가 필요합니다(피크 약 17GB). 16GB P100은 한계에 가깝거나 OOM이 날 수 있지만, 24GB A10G는 안정적입니다.
> - **단점:** Snowflake는 유료이며 크레딧을 소모합니다. 다만 **트라이얼은 30일 동안 $400**를 제공하므로 약 200시간 이상의 GPU 사용이 가능합니다.

---

## ⚠️ 중요한 아키텍처 참고

Snowflake에서는 React 인터페이스 `ace-step-ui`가 아니라 **ACE-Step 네이티브 Gradio 앱**(`app.py`)을 실행합니다.

이유는 **LoRA Training** 탭(원클릭 LoRA 학습)이 네이티브 Gradio 앱에만 있기 때문입니다. 이 앱 하나에서 **생성**과 **LoRA 학습**을 모두 처리합니다.

---

## 0. 필요한 것

- Snowflake 계정. $400 크레딧 트라이얼: https://signup.snowflake.com/
  **AWS eu-west-2 (London)** 리전으로 등록하면 `GPU_NV_S`(A10G) 인스턴스를 사용할 수 있습니다.
- **ACCOUNTADMIN** 권한. 패키지/모델 다운로드와 공개 URL 생성을 위해 External Access Integration을 만들어야 합니다.
  - 💡 트라이얼 계정에서는 기본적으로 사용자가 **ACCOUNTADMIN**입니다.
  - 확인 방법: Snowsight 왼쪽 아래 역할 이름을 눌러 목록에 `ACCOUNTADMIN`이 있으면 전환합니다. 또는 SQL에서 `SELECT CURRENT_AVAILABLE_ROLES();`를 실행합니다.

---

## 1. 인터넷 접근 설정(SQL, 한 번만)

Snowsight에서 **Worksheets → New SQL Worksheet**를 열고 다음을 실행합니다.

```sql
USE ROLE ACCOUNTADMIN;

-- 1) 네트워크 규칙: 모든 호스트로의 outbound 트래픽 허용
--    pip / HuggingFace / 공개 터널에 필요
CREATE OR REPLACE NETWORK RULE acestep_allow_all
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = ('0.0.0.0:443', '0.0.0.0:80');

-- 2) 이 규칙을 사용하는 External Access Integration
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION acestep_eai
  ALLOWED_NETWORK_RULES = (acestep_allow_all)
  ENABLED = TRUE;

-- 3) 시스템 GPU 풀(A10G)과 EAI 사용 권한을 역할에 부여
--    트라이얼에서는 보통 ACCOUNTADMIN 역할 사용
GRANT USAGE ON INTEGRATION acestep_eai TO ROLE ACCOUNTADMIN;
GRANT USAGE ON COMPUTE POOL SYSTEM_COMPUTE_POOL_GPU TO ROLE ACCOUNTADMIN;

-- 4) 선택: 세션 사이에 가중치와 학습된 LoRA를 보존할 stage
CREATE STAGE IF NOT EXISTS ACESTEP_STAGE
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
```

> `SYSTEM_COMPUTE_POOL_GPU` = `GPU_NV_S` 인스턴스 = **1× A10G 24GB, 6 vCPU, 28GB RAM, 450GB NVMe**입니다. 모든 계정에서 제공되며 Container Runtime 노트북에서 바로 사용할 수 있어 별도 풀을 만들 필요가 없습니다.

---

## 2. GPU 노트북 생성(Notebooks on Container Runtime)

1. Snowsight: **Notebooks → + Notebook**.
2. 생성 대화상자:
   - **Runtime / Run on:** `Run on container`(Container Runtime).
   - **Compute type:** `GPU`.
   - **Compute pool:** `SYSTEM_COMPUTE_POOL_GPU (GPU_NV_S)`.
   - **External access integrations:** 1단계에서 만든 **`acestep_eai`** 활성화.
3. **Create** → 컨테이너가 시작될 때까지 기다립니다(상태 Active).

> AWS GPU 노트북은 빠른 NVMe 디스크(~450GB)를 사용하므로 XL 가중치(~20GB)를 저장할 공간이 충분합니다.

---

## 3. 노트북 셀

> 각 셀은 Python입니다. 셸 명령은 `!` 또는 `%%bash`로 실행합니다.

### 셀 1 — GPU와 리소스 확인

```python
!nvidia-smi
!echo '--- RAM ---'; free -h
!echo '--- DISK ---'; df -h /
!python -c "import sys; print('Python', sys.version)"
!python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```

예상 결과: `NVIDIA A10G`, `CUDA: True`, 약 24GB VRAM, 약 28GB RAM. ACE-Step은 **Python ≥ 3.11**을 요구합니다. 이미지가 3.10이면 끝의 참고를 확인하세요.

### 셀 2 — 시스템 의존성(ffmpeg)

```python
# 오디오 처리를 위해 ffmpeg가 필요합니다. Container Runtime에서는 conda(conda-forge 채널)로 설치합니다.
import subprocess, shutil
if shutil.which('ffmpeg') is None:
    subprocess.run('conda install -y -c conda-forge ffmpeg', shell=True)
print('ffmpeg:', shutil.which('ffmpeg'))
```

### 셀 3 — 엔진 클론과 의존성 설치

```python
import os
WORK = '/home/app/acestep'          # 컨테이너 NVMe
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)

# LoRA Training이 포함된 네이티브 ACE-Step 1.5 엔진 클론
if not os.path.isdir('ACE-Step-1.5'):
    !git clone https://github.com/ace-step/ACE-Step-1.5.git
os.chdir('ACE-Step-1.5')

# 사전 설치된 CUDA torch를 망가뜨리지 않도록 패키지 자체만 설치
!pip install -e . --no-deps

# 런타임 의존성 추가 설치(torch 제외) — 엔진에서 확인된 버전 범위
!pip install --no-cache-dir \
  "transformers>=4.51.0,<4.58.0" "diffusers>=0.37.0" "accelerate>=1.12.0" \
  "huggingface_hub[hf_xet]>=0.34.0,<1.0" \
  "soundfile>=0.13.1" librosa soxr loguru einops scipy diskcache numba \
  "vector-quantize-pytorch>=1.27.15" pytorch-wavelets pywavelets toml modelscope matplotlib \
  gradio nano-vllm

!python -c "import torch,transformers; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), '| transformers', transformers.__version__)"
```

> `nano-vllm` 설치가 실패하면 건너뛰어도 됩니다. **LoRA 학습**에는 LM이 필요 없고, 생성에서는 PyTorch 백엔드 LM을 사용할 수 있습니다.

### 셀 4 — 모델 저장 위치(HuggingFace 캐시를 NVMe에)

```python
import os
os.environ['HF_HOME'] = '/home/app/hf'      # 큰 NVMe 디스크
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.makedirs('/home/app/hf', exist_ok=True)
print('HF_HOME =', os.environ['HF_HOME'])
```

`app.py --config acestep-v15-xl-turbo`는 첫 실행 시 HuggingFace에서 XL 가중치(~20GB)를 이 캐시에 자동 다운로드합니다.

### 셀 5 — Gradio 실행 + 공개 URL(cloudflared)

```python
import os, subprocess, time, threading, urllib.request

WORK = '/home/app/acestep/ACE-Step-1.5'
os.chdir(WORK)
env = os.environ.copy()
env['HF_HOME'] = '/home/app/hf'

# 1) 공개 URL 터널용 cloudflared 다운로드
if not os.path.exists('/home/app/cloudflared'):
    !wget -q -O /home/app/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    !chmod +x /home/app/cloudflared

# 2) ACE-Step Gradio 앱을 포트 7860에서 실행
#    --config는 DiT 모델을 지정합니다: acestep-v15-xl-turbo = 4B XL turbo
#    LM은 UI에서 나중에 로드할 수 있으며, LoRA에는 필수가 아닙니다.
log = open('/home/app/gradio.log', 'w')
proc = subprocess.Popen(
    ['python', 'app.py', '--config', 'acestep-v15-xl-turbo', '--port', '7860'],
    cwd=WORK, env=env, stdout=log, stderr=subprocess.STDOUT
)
print('Gradio PID', proc.pid, '— 포트가 열릴 때까지 기다립니다(첫 실행 시 XL 모델 다운로드 5~15분)...')

# 3) localhost:7860이 올라올 때까지 대기
for i in range(180):
    try:
        urllib.request.urlopen('http://localhost:7860', timeout=3); print('Gradio up!'); break
    except Exception:
        time.sleep(5)

# 4) 7860으로 cloudflared 터널 실행
cf = subprocess.Popen(['/home/app/cloudflared','tunnel','--url','http://localhost:7860','--no-autoupdate'],
                       stdout=open('/home/app/cf.log','w'), stderr=subprocess.STDOUT)
time.sleep(8)
!grep -o 'https://[-a-z0-9]*\.trycloudflare\.com' /home/app/cf.log | head -1
print('^ 이 URL을 여세요. 비어 있으면 잠시 기다린 뒤 셀 6을 실행하세요.')
```

### 셀 6 — 공개 URL과 로그 끝부분 표시

```python
!echo '--- PUBLIC URL ---'; grep -o 'https://[-a-z0-9]*\.trycloudflare\.com' /home/app/cf.log | head -1
!echo '--- gradio.log (tail) ---'; tail -n 30 /home/app/gradio.log
```

---

## 4. LoRA 학습 — 단계별(LoRA Training 탭)

공개 URL 열기 → **Initialize Service** 클릭(모델을 메모리에 로드). 아래쪽에 생성 탭과 **LoRA Training** 탭이 표시됩니다.

### 1단계. 데이터셋 준비(자신의 오리지널 트랙)

예를 들어 `/home/app/dataset/` 폴더에 다음 구조로 파일을 둡니다.

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

- **트랙 수:** 스타일 학습에는 8~20개를 권장합니다. 더 많아도 됩니다. 양보다 품질이 중요합니다.
- **BPM/Key:** https://vocalremover.org/key-bpm-finder 에서 쉽게 얻고 Export CSV로 데이터셋 폴더에 넣을 수 있습니다.
- **Caption:** UI의 **Auto Label**(LM 모델 `acestep-5Hz-lm`)로 생성하거나 직접 작성할 수 있습니다.

### 2단계. 스캔과 라벨링

1. **LoRA Training** 탭 → 데이터셋 폴더 경로 입력 → **Scan**.
2. 각 트랙에 가사와 caption이 있는지 확인합니다(**Labeled** 컬럼 = ✅).
3. 필요하면 **Auto Label**을 실행합니다(LM 모델 로드 필요). 항목을 수정하고 수정할 때마다 **Save**를 누릅니다.
4. **Save Dataset**으로 JSON을 내보냅니다.

### 3단계. 텐서 전처리

- 텐서 저장 경로를 지정하고 전처리를 실행합니다.
- 💡 caption 생성을 위해 LM을 사용했다면, VRAM 확보를 위해 **LM 없이 Gradio를 재시작**한 뒤 저장된 JSON을 로드하고 전처리를 실행하세요.

### 4단계. 학습

**LoKr**(탭: **Train LoKr**)을 권장합니다. LoRA보다 약 10배 빠르고(한 시간 대신 몇 분), A10G 한 장에 잘 맞습니다.

1. **Train LoRA** 또는 **Train LoKr** 탭 → 텐서 경로 입력 → 로드.
2. 파라미터(기본값도 대체로 괜찮음):
   - **Max Epochs:** 약 100개 트랙 → 500 epochs, 10~20개 트랙 → 약 800 epochs.
   - **Batch Size:** 1(24GB에서는 2도 시도 가능).
   - **gradient_checkpointing:** OOM이 나면 켭니다(느려지지만 VRAM 절약).
   - **Save Every N Epochs:** 5.
3. **Start Training** → loss 곡선을 확인합니다.

### 5단계. 학습된 LoRA 사용

1. **Gradio를 재시작**하고 모델을 로드합니다(LM 없이).
2. 학습된 LoRA/LoKr 파일을 로드합니다.
3. 자신의 스타일로 음악을 생성합니다 🎶

### 대안: REST API로 학습

엔진은 `localhost:8001`에서 HTTP API를 제공합니다.

```bash
curl -X POST http://localhost:8001/v1/training/start_lokr \
  -H 'Content-Type: application/json' \
  -d '{
    "tensor_dir": "/home/app/tensors",
    "output_dir": "/home/app/lokr_output",
    "lokr_linear_dim": 64,
    "lokr_linear_alpha": 128,
    "lokr_factor": -1,
    "lokr_weight_decompose": true,
    "learning_rate": 0.03,
    "train_epochs": 500,
    "train_batch_size": 1,
    "gradient_accumulation": 4,
    "save_every_n_epochs": 5
  }'
```

LoRA는 `POST /v1/training/start` 엔드포인트와 rank/alpha/dropout 필드를 사용합니다.

---

## 5. 세션 사이 저장(중요)

컨테이너 NVMe 디스크는 **노트북을 중지하면 지워집니다**. XL 가중치를 다시 받지 않고 학습된 LoRA를 잃지 않으려면 `ACESTEP_STAGE` stage에 저장하세요.

```python
from snowflake.snowpark.context import get_active_session
session = get_active_session()

# 학습된 LoRA를 stage에 저장
session.file.put('/home/app/lokr_output/*', '@ACESTEP_STAGE/lora/', auto_compress=False, overwrite=True)

# 다음 세션에서 다시 다운로드
session.file.get('@ACESTEP_STAGE/lora/', '/home/app/lokr_output/')
```

---

## 6. 비용과 크레딧 절약

- `GPU_NV_S`(A10G)는 **대략 시간당 0.5~1 credit** 수준입니다. 정확한 수치는 Snowflake Service Consumption Table을 확인하세요: https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf
  credit당 약 $2~3라면 시간당 약 $1~2입니다.
- 트라이얼 **$400**이면 수백 시간의 GPU 사용량에 해당합니다.
- 작업하지 않을 때는 반드시 세션을 종료하세요. 노트북에서 **End session**을 누릅니다(연결 드롭다운). 유휴 풀도 크레딧을 소모합니다.
- 풀 자동 절전도 설정할 수 있습니다: `ALTER COMPUTE POOL SYSTEM_COMPUTE_POOL_GPU SET AUTO_SUSPEND_SECS = 600;`

---

## 7. 참고와 문제 해결

- **이미지의 Python < 3.11:** ACE-Step은 3.11 이상이 필요합니다. Container Runtime이 3.10이라면 `conda create -y -n ace python=3.11 && conda activate ace`로 conda 환경을 만들고 CUDA용 torch를 다시 설치한 뒤 셀 3을 반복하세요.
- **터널 URL이 비어 있음:** 10~20초 기다린 뒤 셀 6을 다시 실행하세요. cloudflared URL은 즉시 출력되지 않을 수 있습니다.
- **학습 중 OOM:** `gradient_checkpointing`을 켜고, 트랙 길이/데이터셋을 줄이고, LoRA 대신 LoKr을 사용하세요.
- **전처리 중 OOM:** 전처리 전에 LM 모델 없이 Gradio를 재시작하세요(LM이 VRAM을 사용합니다).
- **저작권:** 자신이 만든 오리지널 작품 또는 권리가 있는 자료로만 LoRA를 학습하세요.

---

## TL;DR

1. SQL(1단계): network rule + EAI + grant 생성.
2. `SYSTEM_COMPUTE_POOL_GPU`와 EAI `acestep_eai`를 사용하는 GPU 노트북 생성.
3. 셀 1~6: 엔진 설치, Gradio 실행(`app.py --config acestep-v15-xl-turbo`), 공개 URL 획득.
4. **LoRA Training** 탭 → 데이터셋 → Scan → 전처리 → **Train LoKr** → LoRA 로드 → 생성.
5. LoRA를 stage에 저장하고, 크레딧 절약을 위해 **End session**.
