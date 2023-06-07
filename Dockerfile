FROM nvidia/cuda:11.6.0-base-ubuntu20.04
WORKDIR /training
RUN apt update && \
    apt install --no-install-recommends -y build-essential python3 python3-pip git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install -U bitsandbytes
RUN pip install -U git+https://github.com/huggingface/transformers.git
RUN pip install -U git+https://github.com/huggingface/peft.git
RUN pip install -U git+https://github.com/huggingface/accelerate.git
RUN git clone https://github.com/artidoro/qlora.git
