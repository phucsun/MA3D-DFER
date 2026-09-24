FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    wget \
    nano \
    && rm -rf /var/lib/apt/lists/*


COPY . .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

CMD ["bash"]