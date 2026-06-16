# 🎵 ACE-Step UI — Kaggle (GPU P100)

Запуск **ace-step-ui** на бесплатном GPU Kaggle. Главное отличие от Colab: у Kaggle **~29 ГБ RAM** (против 12 ГБ), поэтому большая модель **XL (4B)** грузится без OOM и без патчей движка.

## 🖥️ Какую GPU выбрать — **P100**
- Движок работает на **одной** GPU (без мультиGPU-шардинга), поэтому в режиме **T4 x2 вторая карта простаивает** — толку от двух T4 нет.
- P100 (16 ГБ, ~732 ГБ/с) против одной T4 (16 ГБ, ~320 ГБ/с): диффузия упирается в пропускную способность памяти, у P100 она вдвое выше → генерация быстрее.
- 16 ГБ VRAM хватает: XL в fp16 занимает ~10 ГБ.

## Подготовка (один раз)
1. Нужен аккаунт Kaggle с **подтверждённым телефоном** — иначе не включить Интернет и GPU.
2. **Create → New Notebook**.
3. Панель справа → **Settings → Accelerator → GPU P100**.
4. **Settings → Internet → On** (обязательно).
5. Вставляйте ячейки ниже по порядку и выполняйте сверху вниз (Shift+Enter).
6. В конце откройте публичную ссылку `*.trycloudflare.com`.

## 🎯 Выбор модели (ячейка 6, переменная `DIT_MODEL`)
- `acestep-v15-turbo` — 2B, быстрая (8 шагов), качается автоматически, влезает и в /kaggle/working.
- `acestep-v15-xl-turbo` — **XL 4B, макс. качество при 8 шагах**.
- `acestep-v15-xl-sft` — XL 4B, 50 шагов + CFG (абсолютный максимум, медленно).

> ⚠️ **Важно про диск:** `/kaggle/working` ограничена ~20 ГБ (Output). Веса XL (~20 ГБ) туда не влезают, поэтому ячейка 6 качает их в `/kaggle/temp` (scratch ~50–60 ГБ) и подключает через симлинк.

---

### 0) Проверка GPU и RAM
```python
!nvidia-smi
!free -h
```

### 1) Node.js 20 + ffmpeg + cloudflared
```python
!curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null 2>&1
!apt-get install -y nodejs build-essential ffmpeg > /dev/null 2>&1
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -O /tmp/cf.deb && dpkg -i /tmp/cf.deb > /dev/null 2>&1
!node -v && cloudflared --version
```

### 2) Клонирование UI и движка + установка движка
```python
%cd /kaggle/working
![ -d ace-step-ui_CPU ] || git clone -q https://github.com/Landers125/ace-step-ui_CPU.git
![ -d ACE-Step-1.5 ] || git clone -q https://github.com/ace-step/ACE-Step-1.5.git
%cd /kaggle/working/ACE-Step-1.5
!pip install -q -e . --no-deps
!pip install -q -e acestep/third_parts/nano-vllm --no-deps
!pip install -q "transformers>=4.51.0,<4.58.0" "diffusers>=0.37.0" "accelerate>=1.12.0" "soundfile>=0.13.1" loguru einops scipy "vector-quantize-pytorch>=1.27.15" diskcache numba pytorch-wavelets pywavelets toml modelscope matplotlib librosa soxr
import torch, transformers; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| transformers', transformers.__version__)
```
> Предупреждения pip про `lightning`/`gradio`/`torchao`/RAPIDS — нормально (мы ставим движок с `--no-deps`). Главное — `CUDA: True`.
> Если `CUDA: False`, выполните один раз: `!pip install -q --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu121`

### 3) Зависимости интерфейса
```python
%cd /kaggle/working/ace-step-ui_CPU
!npm install --silent
!cd server && npm install --silent
print('UI deps installed')
```

### 4) Патчи: Vite allowedHosts, CORS (dev), выбор DiT-модели
```python
import pathlib
base = '/kaggle/working/ace-step-ui_CPU'
vc = pathlib.Path(base+'/vite.config.ts'); s = vc.read_text()
if 'allowedHosts' not in s:
    s = s.replace('port: 3000,', 'port: 3000,'+chr(10)+'      allowedHosts: true,'); vc.write_text(s)
print('allowedHosts:', 'allowedHosts' in vc.read_text())
ix = pathlib.Path(base+'/server/src/index.ts'); t = ix.read_text()
needle = "if (config.nodeEnv === 'development') {"
if 'dev-allow-all' not in t:
    t = t.replace(needle, needle + ' return callback(null, true); // dev-allow-all', 1); ix.write_text(t)
print('CORS:', 'dev-allow-all' in ix.read_text())
sg = pathlib.Path(base+'/server/scripts/simple_generate.py'); g = sg.read_text()
if 'ACESTEP_CONFIG_PATH' not in g:
    g = g.replace('config_path="acestep-v15-turbo",', 'config_path=os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-turbo"),'); sg.write_text(g)
print('DiT switchable:', 'ACESTEP_CONFIG_PATH' in sg.read_text())
```

### 5) Публичный туннель cloudflared (порт 3000)
```python
import subprocess, time, re
subprocess.Popen('cloudflared tunnel --url http://localhost:3000 --no-autoupdate > /kaggle/working/cf.log 2>&1', shell=True)
url = None
for _ in range(40):
    time.sleep(2)
    try: log = open('/kaggle/working/cf.log').read()
    except Exception: log = ''
    m = re.search('https://[a-z0-9-]+[.]trycloudflare[.]com', log)
    if m: url = m.group(0); break
print('PUBLIC URL:', url or 'see /kaggle/working/cf.log')
```

### 6) Выбор и скачивание модели (веса XL → /kaggle/temp через симлинк)
```python
import os, shutil, subprocess, sys
DIT_MODEL = 'acestep-v15-xl-turbo'   # или 'acestep-v15-turbo' (2B — маленькая)

# 1) чистим возможную оборванную докачку из /kaggle/working (лимит ~20 ГБ)
shutil.rmtree('/kaggle/working/ACE-Step-1.5/checkpoints', ignore_errors=True)

# 2) большие веса держим в /kaggle/temp (~50–60 ГБ), в движок кладём симлинк
SCRATCH = '/kaggle/temp/checkpoints'; os.makedirs(SCRATCH, exist_ok=True)
LINK = '/kaggle/working/ACE-Step-1.5/checkpoints'
if os.path.islink(LINK):
    os.remove(LINK)
elif os.path.exists(LINK):
    shutil.rmtree(LINK)
os.symlink(SCRATCH, LINK)
print('checkpoints ->', os.path.realpath(LINK))

# 3) свободное место в scratch (для XL нужно ~21 ГБ)
_t,_u,_f = shutil.disk_usage('/kaggle/temp'); print('scratch free: %.1f GB' % (_f/1e9))

# 4) качаем веса
XL_REPOS = {
  'acestep-v15-xl-turbo': 'ACE-Step/acestep-v15-xl-turbo',
  'acestep-v15-xl-base':  'ACE-Step/acestep-v15-xl-base',
  'acestep-v15-xl-sft':   'ACE-Step/acestep-v15-xl-sft',
}
if DIT_MODEL in XL_REPOS:
    subprocess.run([sys.executable,'-m','pip','install','-q','-U','huggingface_hub[hf_xet]','hf_xet'], check=False)
    from huggingface_hub import snapshot_download
    dest = os.path.join(LINK, DIT_MODEL)
    print('Скачиваю', XL_REPOS[DIT_MODEL], '->', os.path.realpath(dest))
    snapshot_download(repo_id=XL_REPOS[DIT_MODEL], local_dir=dest, max_workers=4)
    subprocess.run([sys.executable,'-m','pip','install','-q','huggingface_hub>=0.34.0,<1.0'], check=False)
    print('Готово, файлов:', len(os.listdir(dest)))
else:
    print('2B-модель скачается автоматически при первой генерации.')
```

### 7) .env + запуск backend и frontend
```python
import sys, os, subprocess, time
root = '/kaggle/working/ace-step-ui_CPU'; ENGINE = '/kaggle/working/ACE-Step-1.5'
public = url if ('url' in dir() and url) else 'http://localhost:3000'
open(root+'/.env','w').write(chr(10).join([
  'NODE_ENV=development','PORT=3001','FRONTEND_URL='+public,
  'ACESTEP_API_URL=http://localhost:8001','ACESTEP_PATH='+ENGINE,
  'ACESTEP_CONFIG_PATH='+DIT_MODEL,'PYTHON_PATH='+sys.executable,'']))
e = os.environ.copy(); e['ACESTEP_CONFIG_PATH']=DIT_MODEL; e['ACESTEP_PATH']=ENGINE
subprocess.Popen('npx tsx src/index.ts > /kaggle/working/backend.log 2>&1', shell=True, cwd=root+'/server', env=e)
subprocess.Popen('npm run dev > /kaggle/working/frontend.log 2>&1', shell=True, cwd=root, env=e)
print('Запуск (модель:', DIT_MODEL, '), ждём 30 сек...'); time.sleep(30)
print(open('/kaggle/working/backend.log').read()[-1500:]); print('OPEN:', public)
```

### 8) Сбор готовых треков в одну папку
```python
import threading, time, shutil, os, glob, datetime
AUDIO_DIR = '/kaggle/working/ace-step-ui_CPU/server/public/audio'
OUT_DIR = '/kaggle/working/ACE-Step-Output'; os.makedirs(OUT_DIR, exist_ok=True)
def _sync():
    while True:
        try:
            for f in glob.glob(os.path.join(AUDIO_DIR,'*')):
                if os.path.isfile(f) and f.lower().endswith(('.mp3','.flac','.wav')):
                    day = datetime.date.fromtimestamp(os.path.getmtime(f)).isoformat()
                    dd = os.path.join(OUT_DIR, day); os.makedirs(dd, exist_ok=True)
                    dest = os.path.join(dd, os.path.basename(f))
                    if not os.path.exists(dest): shutil.copy2(f, dest); print('Saved:', day+'/'+os.path.basename(f))
        except Exception: pass
        time.sleep(15)
threading.Thread(target=_sync, daemon=True).start()
print('Треки складываются в', OUT_DIR, '— скачать через правую панель (Output/Data).')
```

---

## ✅ Готово
- Откройте ссылку `https://....trycloudflare.com` из ячейки 5/7.
- XL качает ~20 ГБ весов при первом запуске; на ~29 ГБ RAM грузится без OOM.
- На P100 (16 ГБ VRAM) для XL держите **Batch Size = 1**, длительность 30–120 сек.
- Готовые треки: `/kaggle/working/ACE-Step-Output/ГГГГ-ММ-ДД` — скачиваются через панель Output.
- Логи генерации: `!tail -n 60 /kaggle/working/backend.log`.
- ⚠️ Веса XL лежат в `/kaggle/temp` (scratch, не входит в лимит Output), но **он очищается при завершении сессии** — XL придётся качать заново при каждом новом запуске (либо сохраните их в приватный Kaggle Dataset).
