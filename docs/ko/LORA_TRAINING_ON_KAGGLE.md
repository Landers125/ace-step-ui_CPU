# 🎸 Kaggle에서 LoRA 학습하기(16GB, 무료) — ACE-Step 1.5

이 가이드는 무료 Kaggle GPU **P100/T4 16GB**에서 ACE-Step용 **자체 LoRA**를 학습하는 방법을 설명합니다. 이미 Kaggle 환경을 설정했고 전화 인증도 완료된 상황을 전제로 합니다.

> **⚠️ 16GB에 대한 현실적인 설명:** ACE-Step LoRA 학습에는 **최소 수준**입니다(피크 약 17GB, 권장 20GB+). XL(4B)에서는 OOM이 발생할 수 있으므로 아래에는 메모리 절약 방법(섹션 2)과 16GB에 확실히 들어가는 **2B 모델 대안**(섹션 6)을 함께 제공합니다.

> **아키텍처 참고:** LoRA 학습에는 React `ace-step-ui`가 아니라 **ACE-Step 네이티브 Gradio 앱**(`app.py`)을 실행합니다. **LoRA Training** 탭은 이 앱에만 있습니다. `RUN_ON_KAGGLE_GPU.md`는 생성용 React 인터페이스 실행 문서이므로 별도입니다.

---

## 0. Kaggle 노트북 준비

1. https://kaggle.com → **Create → New Notebook**.
2. 오른쪽 **Settings**:
   - **Accelerator** = `GPU P100` 또는 `GPU T4 x2`(실제로는 GPU 1개만 사용).
   - **Internet** = `On`(전화 인증 필요).
3. ⚠️ **저장소:** `/kaggle/temp`는 세션 사이에 지워집니다. `/kaggle/working`은 노트북 Output으로 보존됩니다. **학습된 LoRA는 `/kaggle/working`에 저장**하세요(섹션 5).

---

## 1. 설치 셀

**셀 1 — GPU 확인:**

```python
!nvidia-smi
!python --version
import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

**셀 2 — ffmpeg:**

```python
!apt-get update -qq && apt-get install -y -qq ffmpeg
```

**셀 3 — 엔진 + 의존성(torch 제외, CUDA 손상 방지):**

```python
import os
os.chdir('/kaggle/temp')
!git clone https://github.com/ace-step/ACE-Step-1.5.git
os.chdir('/kaggle/temp/ACE-Step-1.5')
!pip install -e . --no-deps
!pip install --no-cache-dir \
  "transformers>=4.51.0,<4.58.0" "diffusers>=0.37.0" "accelerate>=1.12.0" \
  "huggingface_hub[hf_xet]>=0.34.0,<1.0" \
  "soundfile>=0.13.1" librosa soxr loguru einops scipy diskcache numba \
  "vector-quantize-pytorch>=1.27.15" pytorch-wavelets pywavelets toml modelscope matplotlib gradio
```

**셀 4 — Gradio 실행 + 공개 URL(cloudflared):**

```python
import os, subprocess, time
os.environ['HF_HOME'] = '/kaggle/temp/hf'   # XL은 여기에 다운로드됨(세션 사이에 삭제)
os.chdir('/kaggle/temp/ACE-Step-1.5')

# cloudflared
!wget -q -O /kaggle/temp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x /kaggle/temp/cloudflared

# Gradio를 백그라운드 실행(XL turbo 4B). 16GB에서는 LM을 초기화하지 않음(VRAM 절약)
subprocess.Popen(['python','app.py','--config','acestep-v15-xl-turbo','--port','7860'])
time.sleep(5)
# tunnel
!/kaggle/temp/cloudflared tunnel --url http://localhost:7860 --no-autoupdate
```

출력된 `https://....trycloudflare.com` 링크를 복사해 브라우저에서 엽니다.

---

## 2. 🔑 메모리 절약 방법(16GB에서는 중요)

XL-LoRA를 16GB에 넣으려면 모두 적용하세요.

1. 전처리/학습 전에 **LM을 초기화하지 않습니다**. LM은 Auto Label에만 필요합니다.
2. **`gradient_checkpointing = true`** — activation 메모리를 크게 줄입니다.
3. **`batch_size = 1`**, `gradient_accumulation = 4`.
4. **짧은 클립 15~30초** — 긴 트랙은 VRAM을 더 많이 씁니다.
5. **LoKr** 사용(전체 LoRA 대신) — 더 가볍고 약 10배 빠릅니다.
6. 필요하다면 **XL BF16 가중치**(`marcorez8/acestep-v15-xl-turbo-bf16`, FP32 약 19GB 대신 약 10GB)를 사용합니다.

그래도 OOM이면 섹션 6의 2B 모델을 사용하세요.

---

## 3. 데이터셋(자신의 오리지널 트랙)

파일을 `/kaggle/temp/dataset/`에 넣습니다. 또는 Kaggle Dataset으로 업로드해 연결할 수도 있습니다(섹션 8).

```
dataset/
├── song1.mp3            # 오디오(15~30초로 자르는 것을 권장)
├── song1.lyrics.txt     # 가사(또는 song1.txt)
├── song1.json           # 메타데이터(선택)
└── ...
```

`song1.json`의 모든 필드는 선택입니다: `caption`, `bpm`, `keyscale`, `timesignature`, `language`.

- 스타일 학습에는 **8~20개 트랙**을 권장합니다(양보다 품질).
- BPM/Key는 https://vocalremover.org/key-bpm-finder 에서 얻고 CSV로 내보내 폴더에 넣을 수 있습니다.

---

## 4. LoRA 학습 — UI 단계별 진행

1. **Initialize Service** → **LoRA Training** 탭.
2. **Scan** → `/kaggle/temp/dataset` 경로 입력 → Labeled 상태 확인. 필요하면 **Auto Label**을 실행하고 수정한 뒤 **Save** → **Save Dataset**.
3. **텐서 전처리** → 저장 경로 지정 → 실행. 💡 LM을 사용했다면 VRAM 확보를 위해 LM 없이 Gradio를 재시작하세요.
4. **Train LoKr** → 텐서 경로 → 파라미터:
   - **output_dir = `/kaggle/working/lokr_output`**(보존되도록).
   - Max Epochs: 약 100개 트랙 → 500, 10~20개 → 약 800.
   - Batch 1, gradient_accumulation 4, **gradient_checkpointing ON**.
   - **Start Training** → LoKr은 보통 약 5분.
5. **사용:** Gradio 재시작 → LoRA/LoKr 파일 로드 → 생성 🎶

자세한 튜토리얼: https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/LoRA_Training_Tutorial.md

---

## 5. 💾 학습된 LoRA 저장(`/kaggle/temp`는 지워짐)

다음 중 하나를 선택하세요.

- **간단한 방법:** `/kaggle/working/lokr_output`에 저장합니다. 노트북을 멈춘 뒤에도 Output에 남아 Data/Output 탭에서 다운로드할 수 있습니다.
- **즉시 다운로드:** 스튜디오 파일 매니저 또는 `from IPython.display import FileLink; FileLink('/kaggle/working/lokr_output/...')`.
- **클라우드 업로드:** HuggingFace(`huggingface_hub.upload_folder`)에 올리거나 새 Kaggle Dataset으로 저장합니다(섹션 8).

---

## 6. 🪂 대안: XL이 OOM일 때 2B 모델로 LoRA 학습

2B 모델은 훨씬 가볍고 16GB에 안정적으로 들어갑니다. `--config` 없이 Gradio를 실행하면 기본 2B turbo를 사용합니다.

```python
subprocess.Popen(['python','app.py','--port','7860'])
```

이후 LoRA 단계는 동일합니다. 단점은 XL보다 품질이 낮다는 것이고, 장점은 더 빠르고 OOM 없이 학습된다는 것입니다.

---

## 7. 문제 해결

- **학습 중 CUDA out of memory:** `gradient_checkpointing`을 켜고, 클립 길이를 줄이고, LM을 빼고, BF16을 사용하세요. 마지막 대안은 2B 모델입니다(섹션 6).
- **전처리 중 OOM:** LM 없이 Gradio를 재시작하세요.
- **Kaggle 세션 종료(9시간 제한 / 주 12시간 GPU):** `/kaggle/temp`의 가중치는 삭제됩니다. LoRA는 즉시 `/kaggle/working`에 저장하세요(섹션 5).
- **저작권:** 자신의 오리지널 트랙으로만 학습하세요.

---

## 8. 🔌 Kaggle API: 데이터셋 업로드와 LoRA 저장

트랙을 매 세션마다 다시 올리지 않고, 학습된 LoRA도 잃지 않으려면 Kaggle API를 쓰는 것이 편합니다.

### 설치와 키

1. Kaggle → **Settings / Account → API → Create New Token** → `kaggle.json` 다운로드.
2. 자신의 PC에서 `~/.kaggle/kaggle.json`에 둡니다(Windows: `C:\Users\<name>\.kaggle\kaggle.json`).

```bash
pip install kaggle
```

### 자신의 트랙을 비공개 데이터셋으로 업로드(1회)

```bash
mkdir my_tracks && cp /path/to/tracks/* my_tracks/
kaggle datasets init -p my_tracks            # dataset-metadata.json 생성
# dataset-metadata.json에서 title과 id 수정: "<username>/ace-lora-tracks"
kaggle datasets create -p my_tracks          # 최초 업로드(기본 비공개)
# 이후 업데이트:
kaggle datasets version -p my_tracks -m "update tracks"
```

노트북에서 **Add Input** → 자신의 데이터셋 검색 → `/kaggle/input/ace-lora-tracks/`에 연결됩니다.

### 학습된 LoRA를 데이터셋으로 저장

```bash
kaggle datasets init -p /kaggle/working/lokr_output
# id를 "<username>/my-ace-lora"로 수정 후:
kaggle datasets create -p /kaggle/working/lokr_output
```

### PC로 다운로드

```bash
kaggle datasets download -d <username>/my-ace-lora
```

### ⚠️ 보안

`kaggle.json`은 계정 비밀번호와 같습니다. 저장소나 공개 노트북에 넣지 마세요. 노출됐다면 Account → API → **Expire API Token** 후 새 토큰을 만드세요.

---

## TL;DR

1. Kaggle → New Notebook → GPU P100 + Internet On.
2. 셀 1~4: 엔진 설치 + `app.py --config acestep-v15-xl-turbo` 실행 + cloudflared.
3. 메모리 절약 설정(섹션 2): LM 없이, gradient_checkpointing, batch 1, 짧은 클립, LoKr.
4. **LoRA Training**: 데이터셋 → Scan → 전처리 → Train LoKr(output은 `/kaggle/working`) → 로드 → 생성.
5. `/kaggle/working`에서 LoRA를 저장하세요(섹션 5). 또는 Kaggle API를 사용하세요(섹션 8). XL이 OOM이면 2B 모델을 쓰세요(섹션 6).
