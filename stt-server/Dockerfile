FROM nvcr.io/nvidia/pytorch:25.04-py3

# 시스템 필수 도구
RUN apt update && apt install -y ffmpeg

WORKDIR /app
COPY . /app

# 파이썬 라이브러리 설치
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "stt_api_server.py"]

