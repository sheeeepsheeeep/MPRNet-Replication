# PyTorch image with CUDA 
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# Set the working directory i
WORKDIR /workspace/MPRNet

# Install dependencies 
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries 
RUN pip install --no-cache-dir opencv-python scikit-image tqdm matplotlib \
    && pip install "numpy<2" \
    && pip install "opencv-python<4.9"

# Copy project directory into the container
COPY . /workspace/MPRNet/

# Set the default command 
CMD ["/bin/bash"]