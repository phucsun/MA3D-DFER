# MA3D-Net: Multi-modal Adaptive 3D Morphable Model Modulation Network for Robust Facial Expression Recognition

## Dataset Preparation

Download and prepare the following datasets:

- **[RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset)**
- **[FERPlus](https://github.com/microsoft/FERPlus)**
- **[CAER-S](https://caer-dataset.github.io/)**
- **[CheoFaMo](https://data.mendeley.com/datasets/2h6hxwkbwn/2)**
### Expected Directory Structure

Ensure your dataset is organized as follows:

```
- Datasets/raf-db/
    - test/
        - 1/
            - test_0002_aligned/
                - test_0002_aligned.jpg
                - *.npy (3DMM extracted from EMOCA)
    - train/
```

## Environment Setup

### Option 1: Conda

```bash
conda create -n ma3d python=3.8
conda activate ma3d
pip install -r requirements.txt
```

### Option 2: Docker

Use the provided Dockerfile to build the environment.

## Pretrained Models

Download pretrained weights (Image backbone and Landmark backbone) from [here](https://drive.google.com/drive/folders/1X9pE-NmyRwvBGpVzJOEvLqRPRfk_Siwq?usp=sharing). Place the downloaded files into the `checkpoints` folder.

Expected structure:

```
- checkpoints/
    - ir50.pth
    - mobilefacenet_model_best.pth.tar
```



## Training

To train on a specific dataset (e.g., RAF-DB):


```bash
python train.py --data_type RAF-DB --batch_size 8 --resume_name checkpoints/rafdb_best.pth
```

The `batch_size` can be scaled according to the available GPU resources. Using a larger batch size may lead to better performance. Training logs are saved in the `log` directory. Running the training multiple times is recommended to obtain optimal results.
## Testing

If you want to use our pre-trained model, you can download the best checkpoint [here](https://drive.google.com/drive/folders/10tltNSB009VoalxTo46IUMpQhPANudhF?usp=drive_link) and place it into the `checkpoints/` directory.

To evaluate on a specific dataset (e.g., RAF-DB), run:

```bash
python test.py --data_type RAF-DB --ckpt_path checkpoints/rafdb_best.pth
```

## Visualization

You can visualize model outputs such as:

- t-SNE visualizations  
- Attention maps  
- Radar charts  

Run the scripts in the `scripts/` directory.
