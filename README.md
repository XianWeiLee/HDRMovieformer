# HDRMovieformer

HDRMovieformer: A Transformer Framework and Benchmark for Cinematic SDR-to-HDR Conversion (AAAI-26)

Paper: [aaai2026.pdf](./aaai2026.pdf)

## 1. Create Environment

We use pytorch 2.7.1+cu128.

### 1.1 Install the environment

- Make Conda Environment
```
conda create -n HDRformer python=3.9
conda activate HDRformer
```

- Install Dependencies
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort tensorboard h5py tqdm 

pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
```

- Install BasicSR
```
python setup.py develop --no_cuda_ext
```
   
## 2. Testing
```
python TestHDR/my_test.py
```

you can turn .dpx file into .tiff file using ffmpeg


## 3. Training

```shell
# activate the enviroment
conda activate HDRformer
python3 basicsr/train.py --opt HDR.yml
```
