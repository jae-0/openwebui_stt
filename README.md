# 전북특별자치도청 오픈소스 챗봇 - 오디오 STT 파이프라인

전북특별자치도청의 업무 요구사항에 맞게 Open WebUI를 커스터마이징한 AI 업무 보조 시스템의 핵심 STT(Speech-to-Text) 모듈입니다.
오디오/영상 파일을 업로드하면 자동으로 고정밀 STT 처리가 수행되며, 긴 오디오 통째 입력 시 발생하는 OOM(Out of Memory) 및 환각(Hallucination) 문제를 청크 분할과 병렬 처리로 완벽히 해결했습니다.

---

## 🎯 주요 기능 및 특징

- **자동화된 파이프라인**: 오디오/영상 파일 업로드 시 즉시 백그라운드 변환 시작
- **OOM 및 Hallucination 해결**: 무음 구간 기반 청크 분할(`split_on_silence`) 알고리즘을 통한 장시간 파일 안정적 지원
- **고정밀/고효율 음성 인식**: `openai/whisper-large-v3` 모델 연동 (HuggingFace `transformers.pipeline` 활용)
- **동적 병렬 처리 최적화**: `ThreadPoolExecutor` 기반 병렬 처리 (배치 크기 4, GPU/CPU 동적 워커 수 조정)
- **하드웨어 자동 전환**: GPU(CUDA, fp16) 및 CPU(fp32) 환경 자동 감지 및 최적화 실행
- **이중 구조 설계**: `faster-whisper` 기반 단순 단문 STT와 장시간 처리용 파이프라인 분리
- **즉각적인 UI 연동**: 생성된 `.txt` 결과물을 Storage에 업로드하고, Files DB 테이블에 메타데이터를 자동 등록하여 UI에서 즉시 접근 가능

---

## 🛠 기술 스택 (Tech Stack)

- **Language**: Python
- **Framework & DB**: FastAPI, SQLAlchemy, Open WebUI
- **AI & ML**: PyTorch, HuggingFace Transformers, Whisper large-v3
- **Audio Processing**: pydub

---

## 🔄 처리 흐름

```text
파일 업로드 (audio/*, video/webm)
         │
         ▼
     오디오 전처리 (pydub 활용)
   (WAV 변환, 16kHz 리샘플링, 모노 변환, 노말라이즈, +6dB 볼륨 부스트)
         │
         ▼
   무음 구간 기반 오디오 청크 분할
   (최대 25초/청크 제한, 2초 미만 짧은 청크는 인접 청크에 병합)
         │
         ▼
   병렬 Whisper 추론 (ThreadPoolExecutor, 배치 크기 4)
   - GPU: fp16, CUDA
   - CPU: fp32 fallback
         │
         ▼
   결과 저장 및 DB 연동
   └── Storage 업로드 → {filename}.txt (전체 텍스트 저장)
   └── Files DB 테이블 메타데이터 등록 (UI 즉시 반영)
```

---

## 📂 핵심 모듈 설명

### `audio.py`

| 함수 | 역할 |
|------|------|
| `convert_to_wav()` | pydub으로 오디오 전처리 (16kHz 모노, 정규화, +6dB 부스트 적용) |
| `split_audio()` | 무음 구간 청크 분할. 최대 25초 제한 및 2초 미만 청크 병합 로직 포함 |
| `process_chunk()` | 개별 청크에 Whisper 파이프라인 실행 후 텍스트 결과 반환 |
| `transcribe_long_audio()` | 전체 파이프라인 조율. 동적 워커 및 배치 단위 병렬 처리 수행 |
| `transcribe()` | 최상위 진입점. 최종 결과를 `.txt`로 저장하고 Storage/DB에 연동 |
| `transcription_handler()` | `faster-whisper` 엔진 사용 시 단순 STT 처리 로직 |

### `files.py` (API 및 DB 연동부)

파일 업로드 시 `content_type`에 따라 분기하며, 전사된 텍스트는 RAG 파이프라인에 인덱싱될 수 있도록 DB에 등록됩니다.

```python
if file.content_type.startswith("audio/") or file.content_type in {"video/webm"}:
    result = transcribe(request, file_path, file_metadata, filedata)
    # Storage 업로드 및 Files DB 테이블에 등록하여 UI에서 즉시 확인 가능하도록 처리
    process_file(request, ProcessFileForm(file_id=id, content=result.get("text", "")))
```

---

## 📄 출력 파일 구조

### `.txt` (일반 텍스트 문서)
청크별로 인식된 텍스트를 하나로 이어붙여 저장한 최종 텍스트 파일입니다. 생성된 파일은 Storage에 업로드되며 원본 오디오 파일과 `transcript_of` 메타데이터로 연결됩니다.

```text
안녕하세요 전북특별자치도청 주간 회의를 시작하겠습니다. 
첫 번째 안건은 이번 달 오픈소스 챗봇 도입 일정에 관한 내용입니다...
```

---

## ⚙️ 오디오 전처리 파라 파라미터

| 파라미터 | 값 | 설명 |
|---|---|---|
| **샘플레이트** | 16,000 Hz | Whisper 최적 입력 크기 |
| **채널** | 1 (모노) | 연산량 감소를 위한 스테레오 병합 |
| **볼륨 부스트** | +6 dB | 음량이 작은 녹음 파일 보정 |
| **무음 최소 길이** | 300 ms | 청크 분할 기준 묵음 시간 |
| **최소 청크 길이** | 2,000 ms | 문맥 유지를 위해 짧은 구간 병합 |
| **최대 청크 길이** | 25,000 ms | OOM 방지를 위한 초과 청크 강제 분할 |
| **동적 배치 크기** | 4 | ThreadPoolExecutor 병렬 처리 큐 사이즈 |

---

## 📦 패키지 의존성 (Dependencies)

```text
fastapi
sqlalchemy
faster-whisper
transformers
torch
pydub
soundfile
tqdm
```
