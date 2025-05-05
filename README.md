# CropCare AI - Plant Disease Detection
## 1. Demo.
https://github.com/user-attachments/assets/1d1ca408-ad35-4909-8293-b3686cbdcfc8

## 2. Installation.
1. Install [Anaconda](https://www.anaconda.com/), Python and `git`.
2. Creating the env and install the requirements.
  ```bash
  git clone https://github.com/huyluongme/cropcare_ai.git

  cd cropcare_ai 

  conda create -n cropcare_ai python=3.8

  conda activate cropcare_ai

  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

  pip install tqdm

  pip install gradio

  pip install pydantic==2.10.6

  ```

## 3. Download Dataset.
  ```bash
  git clone https://github.com/spMohanty/PlantVillage-Dataset.git

  ```

## 4. Train Model.
  ```bash
  python train.py

  ```

## 5. Run Application.
  ```bash
  python app.py

  ```
