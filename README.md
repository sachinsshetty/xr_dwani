## XR _ DWANI



sudo apt update
sudo apt install libgirepository1.0-dev


python3.12 -m venv venv
source venv/bin/activate

pip install -r requirements.txt


python vision_ai.py



vllm serve RedHatAI/gemma-3-27b-it-FP8-dynamic --served-model-name gemma3 --host 0.0.0.0 --port 9000 --gpu-memory-utilization 0.6 --tensor-parallel-size 1 --max-model-len 32768 --disable-log-requests --dtype bfloat16 --enable-chunked-prefill --enable-prefix-caching --max-num-batched-tokens 8192 --chat-template-content-format openai


--

pip install supervision transformers

pip install torch torchvision opencv-python git+https://github.com/facebookresearch/segment-anything.git

wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth