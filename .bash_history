ls
docker ps -a
cd
ls
docker logs yjy
free -h
nvidia-smi
export NCCL_P2P_DISABLE=1
export NCCL_IGNORE_DISABLED_P2P=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -m vllm.entrypoints.openai.api_server   --model /models/Llama-3.3-70B-Instruct-AWQ   --tensor-parallel-size 2   --gpu-memory-utilization 0.95   --max-model-len 32768   --host 0.0.0.0   --port 8000
curl http://localhost:8000/v1/models
curl http://224.124.155.35:8000/v1/models
ls
docker build -t stt_test_ver1.2 .
ls
cd stt-
ls
cd stt-test
ls
sudo docker run -d -p 3003:8080 -v open-webui:/app/backend/data --name yjy --network=ollama-net --restart always --env ENABLE_SIGNUP=true --env ENABLE_LOGIN_FORM=true --env WEBUI_AUTH=true --env OPENAI_API_BASE_URL=http://stt_test:8000/v1 --env OPENAI_API_KEY=token-abc123 --env ENABLE_OLLAMA_API=true --env OLLAMA_BASE_URL=http://ollama:11434 --env ENABLE_RAG_WEB_SEARCH=true --env RAG_WEB_SEARCH_ENGINE=duckduckgo stt_test_ver1.3
docker exec -it yjy /bin/bash
docker log yjy
docker logs yjy
ls
rm -rf audio.py
nano audio.py
docker build -t stt_test_ver1.4 .
docker ps -a
docker stop yjy
docker rm yjy
sudo docker run -d -p 3003:8080 -v open-webui:/app/backend/data --name yjy --network=ollama-net --restart always --env ENABLE_SIGNUP=true --env ENABLE_LOGIN_FORM=true --env WEBUI_AUTH=true --env OPENAI_API_BASE_URL=http://stt_test:8000/v1 --env OPENAI_API_KEY=token-abc123 --env ENABLE_OLLAMA_API=true --env OLLAMA_BASE_URL=http://ollama:11434 --env ENABLE_RAG_WEB_SEARCH=true --env RAG_WEB_SEARCH_ENGINE=duckduckgo stt_test_ver1.4
docker exec -it yjy /bin/bash
docker logs yjy
ls
rm -rf audio.py
nano audio.py
docker build -t stt_test_ver1.5 .
sudo docker run -d -p 3003:8080 -v open-webui:/app/backend/data --name yjy --network=ollama-net --restart always --env ENABLE_SIGNUP=true --env ENABLE_LOGIN_FORM=true --env WEBUI_AUTH=true --env OPENAI_API_BASE_URL=http://stt_test:8000/v1 --env OPENAI_API_KEY=token-abc123 --env ENABLE_OLLAMA_API=true --env OLLAMA_BASE_URL=http://ollama:11434 --env ENABLE_RAG_WEB_SEARCH=true --env RAG_WEB_SEARCH_ENGINE=duckduckgo stt_test_ver1.5
docker stop yjy
docker rm yjy
sudo docker run -d -p 3003:8080 -v open-webui:/app/backend/data --name yjy --network=ollama-net --restart always --env ENABLE_SIGNUP=true --env ENABLE_LOGIN_FORM=true --env WEBUI_AUTH=true --env OPENAI_API_BASE_URL=http://stt_test:8000/v1 --env OPENAI_API_KEY=token-abc123 --env ENABLE_OLLAMA_API=true --env OLLAMA_BASE_URL=http://ollama:11434 --env ENABLE_RAG_WEB_SEARCH=true --env RAG_WEB_SEARCH_ENGINE=duckduckgo stt_test_ver1.5
docker exec -it yjy /bin/bash
docker logs yjy
ls
rm -rf audio.py
nano audio.py
docker build -t stt_test_ver1.6 .
docker stop yjy
docker rm yjy
sudo docker run -d -p 3003:8080 -v open-webui:/app/backend/data --name yjy --network=ollama-net --restart always --env ENABLE_SIGNUP=true --env ENABLE_LOGIN_FORM=true --env WEBUI_AUTH=true --env OPENAI_API_BASE_URL=http://stt_test:8000/v1 --env OPENAI_API_KEY=token-abc123 --env ENABLE_OLLAMA_API=true --env OLLAMA_BASE_URL=http://ollama:11434 --env ENABLE_RAG_WEB_SEARCH=true --env RAG_WEB_SEARCH_ENGINE=duckduckgo stt_test_ver1.6
docker exec -it yjy /bin/bash
docker stop yjy
docker rm yjy
sudo docker run -p 3003:8080 -v open-webui:/app/backend/data --name yjy --network=ollama-net --restart always --env ENABLE_SIGNUP=true --env ENABLE_LOGIN_FORM=true --env WEBUI_AUTH=true --env OPENAI_API_BASE_URL=http://vllm_yjy:8000/v1 --env OPENAI_API_KEY=token-abc123 --env ENABLE_OLLAMA_API=true --env OLLAMA_BASE_URL=http://ollama:11434 --env ENABLE_RAG_WEB_SEARCH=true --env RAG_WEB_SEARCH_ENGINE=duckduckgo stt_test_ver1.6
docker stop yjy
docker rm yjy
sudo docker run --gpus all -p 3003:8080 -v open-webui:/app/backend/data --name yjy --network=ollama-net --restart always --env ENABLE_SIGNUP=true --env ENABLE_LOGIN_FORM=true --env WEBUI_AUTH=true --env OPENAI_API_BASE_URL=http://vllm_yjy:8000/v1 --env OPENAI_API_KEY=token-abc123 --env ENABLE_OLLAMA_API=true --env OLLAMA_BASE_URL=http://ollama:11434 --env ENABLE_RAG_WEB_SEARCH=true --env RAG_WEB_SEARCH_ENGINE=duckduckgo stt_test_ver1.6
ls
rm -rf audio.py
nano audio.py
docker build -t stt_test_ver1.7 .
docker build -t stt_test_ver1.8 .
cat requirements.txt 
docker ps
docker exec -it vllm /bin/bash
ls
nano requirements.txt 
docker exec -it vllm /bin/bash
ls
nano requirements.txt 
docker build -t stt_test_ver1.9 .
sudo docker run --gpus all -p 3003:8080 -v open-webui:/app/backend/data --name yjy --network=ollama-net --restart always --env ENABLE_SIGNUP=true --env ENABLE_LOGIN_FORM=true --env WEBUI_AUTH=true --env OPENAI_API_BASE_URL=http://vllm_yjy:8000/v1 --env OPENAI_API_KEY=token-abc123 --env ENABLE_OLLAMA_API=true --env OLLAMA_BASE_URL=http://ollama:11434 --env ENABLE_RAG_WEB_SEARCH=true --env RAG_WEB_SEARCH_ENGINE=duckduckgo stt_test_ver1.9
docker stop yjy
docker rm yjy
sudo docker run --gpus all -p 3003:8080 -v open-webui:/app/backend/data --name yjy --network=ollama-net --restart always --env ENABLE_SIGNUP=true --env ENABLE_LOGIN_FORM=true --env WEBUI_AUTH=true --env OPENAI_API_BASE_URL=http://vllm_yjy:8000/v1 --env OPENAI_API_KEY=token-abc123 --env ENABLE_OLLAMA_API=true --env OLLAMA_BASE_URL=http://ollama:11434 --env ENABLE_RAG_WEB_SEARCH=true --env RAG_WEB_SEARCH_ENGINE=duckduckgo stt_test_ver1.9
docker ps -a
docker exec -it open-webui /bin/bash
docker restart open-webui 
docker exec -it open-webui
docker logs open-webui 
docker logs -f open-webui 
docker ps -a
docker
docker ps
ls
docker logs -f open-webui 
docker exec -it open-webui /bin/bash
clear
docker logs -f open-webui 
docker restart open-webui 
docker logs -f open-webui 
ls
cd
ls
cd stt-test
ls
docker ps -a
docker exec -it open-webui_test /bin/bash
docker cp stt_test:/app/backend/open_webui/env.py .
docker cp open_webui-test:/app/backend/open_webui/env.py .
docker ps -a
docker cp open-webui_test:/app/backend/open_webui/env.py .
ls
docker exec -it open-webui_test /bin/bash
ls
nano Dockerfile 
rm -rf audio_test.py 
docker build -t stt_test_ver1.0 .
docker ps -a
docker stop open-webui_test
docker rm open-webui_test
sudo docker run -d -p 3003:8080 -v open-webui:/app/backend/data --name open-webui_test --network=ollama-net --restart always --env ENABLE_SIGNUP=true --env ENABLE_LOGIN_FORM=true --env WEBUI_AUTH=true --env OPENAI_API_BASE_URL=http://stt_test:8000/v1 --env OPENAI_API_KEY=token-abc123 --env ENABLE_OLLAMA_API=true --env OLLAMA_BASE_URL=http://ollama:11434 --env ENABLE_RAG_WEB_SEARCH=true --env RAG_WEB_SEARCH_ENGINE=duckduckgo stt_test_ver1.0
docker exec -it open-webui_test /bin/bash
docker ps -a
docker restart open-webui_test 
docker exec -it open-webui_test /bin/bash
docker rm open-webui_Test
docker rm open-webui_test
docker stop open-webui_test
docker rm open-webui_test
sudo docker run -d -p 3003:8080 -v open-webui:/app/backend/data --name open-webui_test --network=ollama-net --restart always --env ENABLE_SIGNUP=true --env ENABLE_LOGIN_FORM=true --env WEBUI_AUTH=true --env OPENAI_API_BASE_URL=http://stt_test:8000/v1 --env OPENAI_API_KEY=token-abc123 --env ENABLE_OLLAMA_API=true --env OLLAMA_BASE_URL=http://ollama:11434 --env ENABLE_RAG_WEB_SEARCH=true --env RAG_WEB_SEARCH_ENGINE=duckduckgo stt_test_ver1.0
docker exec -it open-webui_test /bin/bash
nvidia-smi
docker logs open-webui_test
ls
docker ps -a
docker exec -it open-webui /bin/bash
docker cp open_webui:/app/backend/open_webui/routers/audio.py .
docker cp open-webui:/app/backend/open_webui/routers/audio.py .
ls
nano audio.py
ls
docker build -t stt_test_ver1.1 .
docker ps -a
docker stop open-webui_test
docker rm open-webui_test
sudo docker run -d -p 3003:8080 -v open-webui:/app/backend/data --name yjy --network=ollama-net --restart always --env ENABLE_SIGNUP=true --env ENABLE_LOGIN_FORM=true --env WEBUI_AUTH=true --env OPENAI_API_BASE_URL=http://stt_test:8000/v1 --env OPENAI_API_KEY=token-abc123 --env ENABLE_OLLAMA_API=true --env OLLAMA_BASE_URL=http://ollama:11434 --env ENABLE_RAG_WEB_SEARCH=true --env RAG_WEB_SEARCH_ENGINE=duckduckgo stt_test_ver1.1
docker exec -it yjy /bin/bash
ls
nano audio.py
vi audio.py
ls
docker logs yjy
docker exec -it yjy /bin/bash
nvidia-smi
ls
docker exec -it yjy /bin/bash
ls
nano requirements.txt 
docker build -t stt_test_ver1.7 .
LS
ls
ls
cd ..
ls
cd yoon
docker ps -a
sudo docker ps -a
sudo sudo su
sudo docker ps -a
sudo su
sudo sudo su
sudo root
ls
cd ..
ls
docker
docker ps
docker ps -a
su
ls
cd
ls
docker ps -a
su
1
ls
cd ..
ls
cd ..
ls
cd
ls
cd stt-test
ls
vim audio.py
sudo apt-get install nano
nano audio.p
nano audio.py
nano audio.py
nano audio.py
nano audio.py
cp audio.py audio_test.py
nano audio_test.py 
nano audio_test.py 
docker ps -a
ocker ps -a
docker ps -a
docker exec -it open-webui_test /bin/bash
cat /etc/os-release
ls
cat audio.py
nano audio.py
vim audio.py
ls
vim main.py
ls
nvidia-smi
docker ps -a
docker exec -it stt_test /bin/bash
docker exec -it stt_test /bin/bash
ls
docker ps -a
docker exec -it yjy /bin/bash
ls
nano env.py
vi env.py
rm -rf audio.py
vi audio.py
ls
vi audio.py
rm -rf audio.py 
nano audio.py
ls
nvidia-smi
ls
nano requirements.txt
nano Dockerfile
nano requirements.txt 
docker build -t stt_test_ver1.3 .
ls
docker ps -a
docker stop yjy
docker rm yjy
nvidia-smi
docker ps
docker stop vllm
nvidia-smi
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name vllm_yjy --network=ollama-net -p 8008:8000 -v /home/models:/models -it jeonjiho817/vllm-docker-jeon:v2 /bin/bash
ls
sudo docker ps -a
docker exec -it 587e6141ac49 /bash
sudo docker exec -it 587e6141ac49 bash
docker ps -a
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name vllm --network=ollama-net -p 8008:8000 -v /home/models:/models -it jeonjiho817/vllm-docker-jeon:v2 /bin/bash
sudo 
sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name vllm --network=ollama-net -p 8008:8000 -v /home/models:/models -it jeonjiho817/vllm-docker-jeon:v2 /bin/bash
sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name stt_test --network=ollama-net -p 8008:8000 -v /home/models:/models -it jeonjiho817/vllm-docker-jeon:v2 /bin/bash
ls
docker ps -a
sudo docker restart 3d6843f8f89d
docker exec -it 3d6843f8f89d bash
docker ps -a
docker run -d -p 3000:8080 -v open-webui:/app/backend/data --name open-webui_test \ 
--network=ollama-net --restart always --env ENABLE_SIGNUP=true --env ENABLE_LOGIN_FORM=true --env WEBUI_AUTH=true --env OPENAI_API_BASE_URL=http://stt_test:8000/v1 --env OPENAI_API_KEY=token-abc123 --env ENABLE_OLLAMA_API=true --env OLLAMA_BASE_URL=http://ollama:11434 --env ENABLE_RAG_WEB_SEARCH=true --env RAG_WEB_SEARCH_ENGINE=duckduckgo ghcr.io/open-webui/open-webui:main
sudo docker run -d -p 3000:8080 -v open-webui:/app/backend/data --name open-webui_test \ 
--network=ollama-net --restart always --env ENABLE_SIGNUP=true --env ENABLE_LOGIN_FORM=true --env WEBUI_AUTH=true --env OPENAI_API_BASE_URL=http://stt_test:8000/v1 --env OPENAI_API_KEY=token-abc123 --env ENABLE_OLLAMA_API=true --env OLLAMA_BASE_URL=http://ollama:11434 --env ENABLE_RAG_WEB_SEARCH=true --env RAG_WEB_SEARCH_ENGINE=duckduckgo ghcr.io/open-webui/open-webui:main
sudo docker run -d -p 3000:8080 -v open-webui:/app/backend/data --name open-webui_test --network=ollama-net --restart always --env ENABLE_SIGNUP=true --env ENABLE_LOGIN_FORM=true --env WEBUI_AUTH=true --env OPENAI_API_BASE_URL=http://stt_test:8000/v1 --env OPENAI_API_KEY=token-abc123 --env ENABLE_OLLAMA_API=true --env OLLAMA_BASE_URL=http://ollama:11434 --env ENABLE_RAG_WEB_SEARCH=true --env RAG_WEB_SEARCH_ENGINE=duckduckgo ghcr.io/open-webui/open-webui:main
docker run -d -p 3003:8080 \ 
-v open-webui:/app/backend/data --name open-webui_test --network=ollama-net --restart always --env ENABLE_SIGNUP=true --env ENABLE_LOGIN_FORM=true --env WEBUI_AUTH=true --env OPENAI_API_BASE_URL=http://stt_test:8000/v1 --env OPENAI_API_KEY=token-abc123 --env ENABLE_OLLAMA_API=true --env OLLAMA_BASE_URL=http://ollama:11434 --env ENABLE_RAG_WEB_SEARCH=true --env RAG_WEB_SEARCH_ENGINE=duckduckgo ghcr.io/open-webui/open-webui:main
ls
docker ps -a
docker stop 4a95b3c8662b
docker rm 4a95b3c8662b
sudo docker run -d -p 3003:8080 \ 
-v open-webui:/app/backend/data --name open-webui_test --network=ollama-net --restart always --env ENABLE_SIGNUP=true --env ENABLE_LOGIN_FORM=true --env WEBUI_AUTH=true --env OPENAI_API_BASE_URL=http://stt_test:8000/v1 --env OPENAI_API_KEY=token-abc123 --env ENABLE_OLLAMA_API=true --env OLLAMA_BASE_URL=http://ollama:11434 --env ENABLE_RAG_WEB_SEARCH=true --env RAG_WEB_SEARCH_ENGINE=duckduckgo ghcr.io/open-webui/open-webui:main
sudo docker ps -a
docker ps -a
sudo docker stop 3d6843f8f89d
sudo docker rm 3d6843f8f89d
sudo docker ps -a
sudo docker run -d -p 3003:8080 -v open-webui:/app/backend/data --name open-webui_test --network=ollama-net --restart always --env ENABLE_SIGNUP=true --env ENABLE_LOGIN_FORM=true --env WEBUI_AUTH=true --env OPENAI_API_BASE_URL=http://stt_test:8000/v1 --env OPENAI_API_KEY=token-abc123 --env ENABLE_OLLAMA_API=true --env OLLAMA_BASE_URL=http://ollama:11434 --env ENABLE_RAG_WEB_SEARCH=true --env RAG_WEB_SEARCH_ENGINE=duckduckgo ghcr.io/open-webui/open-webui:main
docker exec -it stt_test bash
docker ps -a
docker exec -it open-webui_test
docker exec -it open-webui_test bash
docker ps -a
docker run -d --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name stt_test --network=ollama-net -p 8008:8000 \ 
-v /home/models:/models jeonjiho817/vllm-docker-jeon:v2 bash -c "export NCCL_P2P_DISABLE=1 && \
         export NCCL_IGNORE_DISABLED_P2P=1 && \
         export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
         python3 -m vllm.entrypoints.openai.api_server \
         --model /models/deepseek-vl2-small \
         --trust-remote-code \
         --tensor-parallel-size 2 \
         --gpu-memory-utilization 0.85 \
         --max-model-len 2048 \
         --host 0.0.0.0 \
         --port 8000"
sudo docker run -d --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name stt_test --network=ollama-net -p 8008:8000 \ 
-v /home/models:/models jeonjiho817/vllm-docker-jeon:v2 bash -c "export NCCL_P2P_DISABLE=1 && \
         export NCCL_IGNORE_DISABLED_P2P=1 && \
         export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
         python3 -m vllm.entrypoints.openai.api_server \
         --model /models/deepseek-vl2-small \
         --trust-remote-code \
         --tensor-parallel-size 2 \
         --gpu-memory-utilization 0.85 \
         --max-model-len 2048 \
         --host 0.0.0.0 \
         --port 8000"
docker ps -a
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name stt_test --network=ollama-net -p 8008:8000 \ 
-v /home/models:/models -it jeonjiho817/vllm-docker-jeon:v2 /bin/bash
sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name stt_test --network=ollama-net -p 8008:8000 \ 
-v /home/models:/models -it jeonjiho817/vllm-docker-jeon:v2 /bin/bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name stt_test --network=ollama-net -p 8008:8000 -v /home/models:/models -it jeonjiho817/vllm-docker-jeon:v2 /bin/bash
ls
docker ps -a
docker restart open-webui_test
curl http://localhost:8008/v1/models
docker exec -it open-webui_test ping stt_test
docker run -it stt_test /bin/bash
docker ps -a
docker exec -it stt_test /bin/bash
docker ps -a
docker restart open-webui_test
curl http://localhost:8008/v1/models
curl http://http://220.124.155.35:8008/v1/models
curl http://220.124.155.35:8008/v1/models
ls
docker ps -a
docker attach open-webui_test /bin/bash
sudo docker attach open-webui_test /bin/bash
sudo docker exec -it open-webui_test /bin/bash
ls
cd 
ls
mkdir stt-test
cd stt-
ls
cd stt-test/
ls
docker cp open-webui_test:/app/backend/open_webui/routers/audio.py ./audio.py\
docker cp open-webui_test:/app/main.py ./main.py
docker cp open-webui_test:/app/backend/main.py ./main.py
ls
docker cp open-webui_test:/app/backend/open_webui/main.py ./main.py
vi main.py 
vi Dockerfile
ls
docker build -t openwetui-custom_jy .
docker stop open-webui_test
docker rm open-webui_test
docker run -d   -p 3003:8080   -v open-webui:/app/backend/data   --name open-webui_test   --network=ollama-net   --restart always   --env ENABLE_SIGNUP=true   --env ENABLE_LOGIN_FORM=true   --env WEBUI_AUTH=true   --env OPENAI_API_BASE_URL=http://stt_test:8000/v1   --env OPENAI_API_KEY=token-abc123   --env ENABLE_OLLAMA_API=true   --env OLLAMA_BASE_URL=http://ollama:11434   --env ENABLE_RAG_WEB_SEARCH=true   --env RAG_WEB_SEARCH_ENGINE=duckduckgo   openwebui-custom
docker ps -a
docker run -d   -p 3003:8080   -v open-webui:/app/backend/data   --name open-webui_test   --network=ollama-net   --restart always   --env ENABLE_SIGNUP=true   --env ENABLE_LOGIN_FORM=true   --env WEBUI_AUTH=true   --env OPENAI_API_BASE_URL=http://stt_test:8000/v1   --env OPENAI_API_KEY=token-abc123   --env ENABLE_OLLAMA_API=true   --env OLLAMA_BASE_URL=http://ollama:11434   --env ENABLE_RAG_WEB_SEARCH=true   --env RAG_WEB_SEARCH_ENGINE=duckduckgo   openwebui-custom_jy
docker ps -a
sudo docker run -d   -p 3003:8080   -v open-webui:/app/backend/data   --name open-webui_test   --network=ollama-net   --restart always   --env ENABLE_SIGNUP=true   --env ENABLE_LOGIN_FORM=true   --env WEBUI_AUTH=true   --env OPENAI_API_BASE_URL=http://stt_test:8000/v1   --env OPENAI_API_KEY=token-abc123   --env ENABLE_OLLAMA_API=true   --env OLLAMA_BASE_URL=http://ollama:11434   --env ENABLE_RAG_WEB_SEARCH=true   --env RAG_WEB_SEARCH_ENGINE=duckduckgo   openwebui-custom_jy
docker images
sudo docker run -d   -p 3003:8080   -v open-webui:/app/backend/data   --name open-webui_test   --network=ollama-net   --restart always   --env ENABLE_SIGNUP=true   --env ENABLE_LOGIN_FORM=true   --env WEBUI_AUTH=true   --env OPENAI_API_BASE_URL=http://stt_test:8000/v1   --env OPENAI_API_KEY=token-abc123   --env ENABLE_OLLAMA_API=true   --env OLLAMA_BASE_URL=http://ollama:11434   --env ENABLE_RAG_WEB_SEARCH=true   --env RAG_WEB_SEARCH_ENGINE=duckduckgo   openwetui-custom_jy
docker ps -a
docker exec -it openwet-ui_jy
sudo docker exec -it openwet-ui_jy
docker ps -a
docker exec -it open-webui_test /bin/bash
ls
cat Dockerfile
docker ps -a
docker exec -it openweb-ui_test
docker exec -it openweb-ui_test /bin/bash
docker ps -a
docker exec -it open-webui_test /bin/bash
ls
vi main.py
vi Dockerfile
docker stop open-webui_test
docker rm open-webui_test
docker images
docker run -d   -p 3003:8080   -v open-webui:/app/backend/data   --name open-webui_test   --network=ollama-net   --restart always   --env ENABLE_SIGNUP=true   --env ENABLE_LOGIN_FORM=true   --env WEBUI_AUTH=true   --env OPENAI_API_BASE_URL=http://stt_test:8000/v1   --env OPENAI_API_KEY=token-abc123   --env ENABLE_OLLAMA_API=true   --env OLLAMA_BASE_URL=http://ollama:11434   --env ENABLE_RAG_WEB_SEARCH=true   --env RAG_WEB_SEARCH_ENGINE=duckduckgo   openwetui-custom_jy 
docker ps -a
docker exec -it open-webui_test /bin/bash
ls
docker build -t openwebui-custom_jy .
docker ps -a
docker stop open-webui_test
docker rm open-webui_test
docker run -d   -p 3003:8080   -v open-webui:/app/backend/data   --name open-webui_test   --network=ollama-net   --restart always   --env ENABLE_SIGNUP=true   --env ENABLE_LOGIN_FORM=true   --env WEBUI_AUTH=true   --env OPENAI_API_BASE_URL=http://stt_test:8000/v1   --env OPENAI_API_KEY=token-abc123   --env ENABLE_OLLAMA_API=true   --env OLLAMA_BASE_URL=http://ollama:11434   --env ENABLE_RAG_WEB_SEARCH=true   --env RAG_WEB_SEARCH_ENGINE=duckduckgo   openwebui-custom_jy
docker exec -it open-webui_test /bin/bash
docker stop open-webui_test
docker rm open-webui_test
docker images
ls
vim Dockerfile
docker build -t openwebui-custom
docker build -t openwebui-custom .
docker run -d   -p 3003:8080   -v open-webui:/app/backend/data   --name open-webui_test   --network=ollama-net   --restart always   --env ENABLE_SIGNUP=true   --env ENABLE_LOGIN_FORM=true   --env WEBUI_AUTH=true   --env OPENAI_API_BASE_URL=http://stt_test:8000/v1   --env OPENAI_API_KEY=token-abc123   --env ENABLE_OLLAMA_API=true   --env OLLAMA_BASE_URL=http://ollama:11434   --env ENABLE_RAG_WEB_SEARCH=true   --env RAG_WEB_SEARCH_ENGINE=duckduckgo   openwebui-custom 
docker exec -it open-webui_test /bin/bash
docker ps -a
ls
cat audio.py
ls
vim audio.py
apt-get install nano
sudo apt-get install nano
su
su
sudo su
sudo whoami
docker ps -a
apt-get install usermod
sudo apt-get install usermod
sudo usermod -aG yoon
sudo usermod -aG docker yoon
docker ps -a
docker ps
sudo docker ps -a
docker ps -a
sudo docker ps -a
docker exec -it 55259e86a871
docker exec -it 55259e86a871 /bash
sudo docker exec -it 55259e86a871 /bash
sudo docker exec -it 55259e86a871 bash
ls
sudo docker ps -a
docker exec -it 587e6141ac49
sudo docker exec -it 587e6141ac49 bash
ls
sudo docker ps -a
ls
cd ..
ls
cd 
ls
mkdir stt-server
ls
cd stt-server/
ls
vi stt_module.py
vi stt_api_server.py
vi requirements.txt
vi Dockerfile
sudo docker ps -a
docker exec -it 587e6141ac49
docker exec -it 587e6141ac49 bash
sudo docker exec -it 587e6141ac49 bash
ls
cd ..
ls
cd ..
ls
cd jeon
ls
sudo su
