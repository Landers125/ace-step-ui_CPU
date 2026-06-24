# 🚀 Google Colab GPU(T4)에서 실행하기

이 방식은 Google Colab의 무료 NVIDIA T4 GPU에서 **ace-step-ui**를 실행합니다. CPU보다 생성 속도가 훨씬 빠릅니다.

## 노트북 열기

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Landers125/ace-step-ui_CPU/blob/main/colab/ACE_Step_UI_GPU_T4.ipynb)

또는 https://colab.research.google.com 에서 `colab/ACE_Step_UI_GPU_T4.ipynb` 파일을 직접 업로드하세요.

## 동작 방식

- ACE-Step 엔진은 PyTorch + CUDA(T4, 16GB VRAM)로 설치됩니다.
- Frontend(Vite, 포트 3000)가 `/api`와 `/audio`를 backend(포트 3001)로 프록시하므로 외부에는 포트 하나만 공개됩니다.
- 공개 링크는 별도 가입 없이 `cloudflared`로 생성됩니다: `https://<...>.trycloudflare.com`.
- 생성은 GPU에서 Python 엔진(`server/scripts/simple_generate.py`)을 직접 호출하는 방식으로 수행됩니다.

## 실행 단계

1. 위 버튼으로 Colab 노트북을 엽니다.
2. **런타임 → 런타임 유형 변경 → T4 GPU**를 선택합니다.
3. 셀을 위에서 아래로 순서대로 실행합니다(Shift+Enter).
4. 마지막 셀에 출력된 공개 링크를 엽니다.

## 제한 사항

- Colab 세션은 임시이며 유휴 상태가 길어지면 종료됩니다. 생성한 트랙은 바로 다운로드하세요.
- 첫 생성 시 모델 가중치(수 GB)를 다운로드합니다.
- 무료 T4에는 일일 사용 시간 제한이 있습니다.
