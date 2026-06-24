# 🎵 ACE-Step — Kaggle(GPU P100) + full-param 웹 인터페이스

무료 Kaggle GPU에서 **ACE-Step** 엔진과 이 프로젝트의 **full-param Flask 인터페이스**(`webui/app.py`, 약 46개 파라미터)를 함께 실행합니다. 모든 것은 Kaggle에서 동작하므로 **PC에 아무것도 설치할 필요가 없고**, 브라우저에서 `*.trycloudflare.com` 링크로 인터페이스를 열 수 있습니다.

준비된 노트북: [`kaggle/ACE_Step_API_Backend_Kaggle.ipynb`](../../kaggle/ACE_Step_API_Backend_Kaggle.ipynb) — Kaggle에서 바로 가져올 수 있습니다(*File → Import Notebook*).

**Kaggle의 장점:** Colab의 약 12GB RAM보다 큰 약 29GB RAM을 제공하므로, 큰 **XL(4B)** 모델도 OOM 없이 엔진 패치 없이 로드됩니다.

## 🧩 아키텍처(백엔드 연결 방식)

- **엔진** ACE-Step은 자체 **REST API**(`acestep-api`)를 `:8001`에서 실행합니다.
- **Flask 인터페이스**(`webui/app.py`)는 `:5000`에서 실행되며 `http://localhost:8001`의 엔진에 접근합니다(`ACE_BASE_URL` 변수).
- 외부에는 **cloudflared**로 `:5000`만 터널링합니다. 브라우저에서는 모든 파라미터를 제공하는 UI를 열게 됩니다.

## 🖥️ 어떤 GPU를 선택할까 — **P100**

- 엔진은 **하나의 GPU**에서 동작합니다(멀티 GPU 샤딩 없음). 따라서 **T4 x2** 모드에서는 두 번째 카드가 놀게 되어 이점이 없습니다.
- P100(16GB, 약 732GB/s)과 단일 T4(16GB, 약 320GB/s)를 비교하면, diffusion은 메모리 대역폭 영향을 크게 받으므로 P100이 더 빠릅니다.
- 16GB VRAM이면 충분합니다. XL fp16은 약 10GB를 사용합니다.

## 준비(한 번만)

1. **전화 인증이 완료된** Kaggle 계정이 필요합니다. 인증이 없으면 Internet과 GPU를 켤 수 없습니다.
2. **Create → New Notebook** 또는 위 링크의 `.ipynb`를 가져옵니다.
3. 오른쪽 패널 → **Settings → Accelerator → GPU P100**.
4. **Settings → Internet → On**(필수).
5. 셀을 위에서 아래로 실행합니다(Shift+Enter).
6. 마지막에 출력되는 `*.trycloudflare.com` 공개 링크를 브라우저에서 엽니다.

## 🎯 모델 선택(셀 3의 `DIT_MODEL` 변수)

- `acestep-v15-turbo` — 2B, 빠름(8 steps), 자동 다운로드.
- `acestep-v15-xl-turbo` — **XL 4B, 8 steps에서 최고 품질**.
- `acestep-v15-xl-sft` — XL 4B, 50 steps + CFG(최고 품질, 느림).

> ⚠️ **디스크 주의:** `/kaggle/working`은 약 20GB(Output)로 제한됩니다. XL 가중치(~20GB)는 여기에 들어가지 않으므로, 셀 3은 `/kaggle/temp`(scratch 약 50~60GB)에 다운로드하고 심링크로 연결합니다.

---

### 0) GPU와 RAM 확인

```python
!nvidia-smi
!free -h
```

### 1) 시스템 패키지 + cloudflared(Node는 더 이상 필요 없음, UI는 Python)

```python
!apt-get install -y ffmpeg > /dev/null 2>&1
# Kaggle에서는 dpkg 방식이 의존성 문제로 실패해 `cloudflared: command not found`가 나오는 경우가 많으므로 바이너리를 PATH에 직접 둡니다.
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
!chmod +x /usr/local/bin/cloudflared
!cloudflared --version
```

> 이전에 `cloudflared: command not found`를 봤다면 예전 `dpkg` 방식 때문입니다. 이제 바이너리를 PATH에 직접 넣습니다.

### 2) 저장소 클론 + 엔진 및 의존성 설치

```python
%cd /kaggle/working
![ -d ace-step-ui_CPU ] || git clone -q https://github.com/Landers125/ace-step-ui_CPU.git
![ -d ACE-Step-1.5 ] || git clone -q https://github.com/ace-step/ACE-Step-1.5.git
%cd /kaggle/working/ACE-Step-1.5
!pip install -q -e . --no-deps
!pip install -q -e acestep/third_parts/nano-vllm --no-deps
!pip install -q "transformers>=4.51.0,<4.58.0" "diffusers>=0.37.0" "accelerate>=1.12.0" "soundfile>=0.13.1" loguru einops scipy "vector-quantize-pytorch>=1.27.15" diskcache numba pytorch-wavelets pywavelets toml modelscope matplotlib librosa soxr python-dotenv
!pip install -q fastapi "uvicorn[standard]" flask requests
import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())
```

> `nano-vllm`은 PyPI에 없고 엔진 저장소의 `acestep/third_parts/nano-vllm`에 있으므로 `-e`로 별도 설치합니다. `lightning`/`gradio`/`torchao` 관련 pip 경고는 정상입니다(엔진을 `--no-deps`로 설치하기 때문). 핵심은 `CUDA: True`입니다.
> `CUDA: False`라면: `!pip install -q --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu121`

### 3) 모델 선택과 다운로드(XL 가중치 → `/kaggle/temp` 심링크)

```python
import os, shutil, subprocess, sys
DIT_MODEL = 'acestep-v15-xl-turbo'   # 또는 'acestep-v15-turbo' (2B)

shutil.rmtree('/kaggle/working/ACE-Step-1.5/checkpoints', ignore_errors=True)
SCRATCH = '/kaggle/temp/checkpoints'; os.makedirs(SCRATCH, exist_ok=True)
LINK = '/kaggle/working/ACE-Step-1.5/checkpoints'
if os.path.islink(LINK):
    os.remove(LINK)
elif os.path.exists(LINK):
    shutil.rmtree(LINK)
os.symlink(SCRATCH, LINK)
print('checkpoints ->', os.path.realpath(LINK))

_t,_u,_f = shutil.disk_usage('/kaggle/temp'); print('scratch free: %.1f GB' % (_f/1e9))

XL_REPOS = {
  'acestep-v15-xl-turbo': 'ACE-Step/acestep-v15-xl-turbo',
  'acestep-v15-xl-base':  'ACE-Step/acestep-v15-xl-base',
  'acestep-v15-xl-sft':   'ACE-Step/acestep-v15-xl-sft',
}
if DIT_MODEL in XL_REPOS:
    subprocess.run([sys.executable,'-m','pip','install','-q','-U','huggingface_hub[hf_xet]','hf_xet'], check=False)
    from huggingface_hub import snapshot_download
    dest = os.path.join(LINK, DIT_MODEL)
    print('다운로드 중', XL_REPOS[DIT_MODEL], '->', os.path.realpath(dest))
    snapshot_download(repo_id=XL_REPOS[DIT_MODEL], local_dir=dest, max_workers=4)
    subprocess.run([sys.executable,'-m','pip','install','-q','huggingface_hub>=0.34.0,<1.0'], check=False)
    print('완료, 파일 수:', len(os.listdir(dest)))
else:
    print('2B 모델은 첫 생성 시 자동으로 다운로드됩니다.')
```

### 4) 엔진 REST API 실행(포트 8001)

```python
import subprocess, os, time
ENGINE = '/kaggle/working/ACE-Step-1.5'
e = os.environ.copy()
e['ACESTEP_API_HOST'] = '0.0.0.0'
e['ACESTEP_API_PORT'] = '8001'
e['ACESTEP_NO_INIT'] = 'true'   # 지연 로딩: 첫 요청 시 UI에서 선택한 모델을 로드
subprocess.Popen('acestep-api > /kaggle/working/engine.log 2>&1', shell=True, cwd=ENGINE, env=e)
print('엔진 REST API가 :8001에서 시작됩니다. 약 40초 기다립니다...'); time.sleep(40)
!tail -n 25 /kaggle/working/engine.log
```

> `acestep-api`는 `ACESTEP_API_HOST`/`ACESTEP_API_PORT`를 읽습니다. `ACESTEP_NO_INIT=true`는 VRAM을 절약합니다. 기본 2B 모델을 XL보다 먼저 로드하지 않습니다. LLM(chain-of-thought)은 기본적으로 꺼져 있어 더 빠르고 최신 torch에서 segfault 위험도 낮습니다.
> `Uvicorn running on http://0.0.0.0:8001` 문구를 기다리세요. 아직 로딩 중이면 `!tail`을 다시 실행합니다.

### 5) full-param 인터페이스 실행(포트 5000)

```python
import subprocess, os, time
UI = '/kaggle/working/ace-step-ui_CPU/webui'
e = os.environ.copy()
e['ACE_BASE_URL'] = 'http://localhost:8001'
e['PORT'] = '5000'
subprocess.Popen('python app.py > /kaggle/working/webui.log 2>&1', shell=True, cwd=UI, env=e)
print('웹 인터페이스가 :5000에서 시작됩니다. 8초 기다립니다...'); time.sleep(8)
!tail -n 15 /kaggle/working/webui.log
```

### 6) cloudflared 공개 터널(포트 5000)

```python
import subprocess, time, re
subprocess.Popen('cloudflared tunnel --url http://localhost:5000 --no-autoupdate > /kaggle/working/cf.log 2>&1', shell=True)
url=None
for _ in range(40):
    time.sleep(2)
    try: log=open('/kaggle/working/cf.log').read()
    except Exception: log=''
    m=re.search('https://[a-z0-9-]+[.]trycloudflare[.]com', log)
    if m: url=m.group(0); break
print('브라우저에서 여세요:', url or '/kaggle/working/cf.log 확인')
```

---

## ✅ 완료

- 셀 6에서 나온 `https://....trycloudflare.com` 링크를 **브라우저에서** 여세요. 이것이 full-param 인터페이스입니다.
- 인터페이스에서 `acestep-v15-xl-turbo` 모델을 선택합니다. 속도를 원하면 `acestep-v15-turbo`를 선택하세요.
- XL은 셀 3에서 약 20GB 가중치를 다운로드합니다. 약 29GB RAM에서는 OOM 없이 로드됩니다.
- P100(16GB VRAM)에서 XL을 쓸 때는 **Batch Size = 1**, 길이는 30~120초를 권장합니다.
- 로그: `!tail -n 60 /kaggle/working/engine.log`(엔진), `!tail -n 60 /kaggle/working/webui.log`(인터페이스).
- ⚠️ XL 가중치는 `/kaggle/temp`에 있습니다(Output 제한에는 포함되지 않는 scratch 공간). 하지만 **세션 종료 시 삭제**되므로 새 실행 때마다 다시 다운로드해야 합니다. 또는 비공개 Kaggle Dataset에 저장하세요.

> 💡 Node `:3000` 기반의 예전 React `ace-step-ui` 방식이 필요하다면 이 파일의 git history를 확인하세요. 현재 버전은 Python 백엔드 `webui/app.py` 기준입니다.
