# Multiclass Fish Image Classification

A small, well-documented project for training and evaluating convolutional neural network models to perform multiclass classification of fish species from images. This repository contains code, model definitions, and instructions to reproduce training, inference, and evaluation on a labeled fish image dataset.

## Table of Contents

- Project overview
- Dataset
- Repository structure
- Requirements
- Installation
- Preparing the data
- Training
- Evaluation and inference
- Model checkpoints
- Results
- Tips & troubleshooting
- Contributing
- License
- Contact

## Project overview

This project demonstrates end-to-end steps for building an image classification pipeline for multiple fish species. It includes data loading and preprocessing, model training (single GPU/CPU), evaluation, and example inference scripts. The code is framework-agnostic but includes examples using PyTorch (recommended) and a simple Keras/TensorFlow outline.

Goals:
- Provide a reproducible training pipeline for multiclass fish classification
- Offer clear scripts for training, validation, and inference
- Document hyperparameters and tips for improving accuracy

## Dataset

This repository expects a dataset organized in the common ImageFolder layout (one folder per class):

dataset/
  train/
    species_1/
      img1.jpg
      img2.jpg
    species_2/
      img1.jpg
  val/
    species_1/
    species_2/
  test/
    species_1/
    species_2/

If you do not have a dataset, you can create a small sample set for testing or download public fish datasets (ensure you have the right to use them). When using a larger dataset, consider image augmentation and balancing classes.

## Repository structure

````markdown
```text
Multiclass-Fish-Image-Classification/
├── data/                      # helper scripts for downloading / preparing dataset (optional)
├── notebooks/                 # Jupyter notebooks for experiments and EDA
├── src/                       # source code (training, models, utils)
│   ├── data.py                # dataset and dataloader utilities
│   ├── train.py               # training loop
│   ├── evaluate.py            # evaluation and metrics
│   ├── infer.py               # inference script
│   └── models/                # model definitions (e.g., resnet, mobilenet)
├── checkpoints/               # where model weights will be saved
├── requirements.txt           # required Python packages
├── README.md                  # this file
└── LICENSE
```
````

## Requirements

A Python 3.8+ environment is recommended.

Typical requirements (also provided in requirements.txt):

- torch
- torchvision
- numpy
- pandas
- scikit-learn
- pillow
- matplotlib
- tqdm
- albumentations (optional, for augmentations)

Install with pip:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository

```bash
git clone https://github.com/Ksh1t1zsharma/Multiclass-Fish-Image-Classification.git
cd Multiclass-Fish-Image-Classification
```

2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Preparing the data

- Place your images in the ImageFolder layout shown above.
- If necessary, write a small script to split images into train/val/test.
- Optionally add data augmentation in `src/data.py` using Albumentations or torchvision.transforms.

## Training

Example command (PyTorch):

```bash
python src/train.py \
  --data-dir ./dataset \
  --model resnet50 \
  --batch-size 32 \
  --epochs 30 \
  --lr 1e-3 \
  --output-dir checkpoints/resnet50
```

Key configuration options (exposed in train.py):
- --data-dir: path to dataset root
- --model: model architecture (resnet18/resnet50/mobilenetv2/...)
- --batch-size
- --epochs
- --lr: learning rate
- --weight-decay: L2 regularization
- --resume: path to checkpoint to resume training
- --num-workers: dataloader workers

Training will save checkpoints and a training log (CSV) in the specified output directory.

## Evaluation and inference

Evaluate a saved checkpoint on the validation or test set:

```bash
python src/evaluate.py --data-dir ./dataset --checkpoint checkpoints/resnet50/best.pth --batch-size 32
```

Run inference on a single image:

```bash
python src/infer.py --checkpoint checkpoints/resnet50/best.pth --image path/to/image.jpg
```

Both scripts will print metrics (accuracy, confusion matrix) and optionally save predictions to a CSV.

## Model checkpoints

- Checkpoints should be stored under `checkpoints/<model-name>/`.
- Save the best model (by validation accuracy) as `best.pth` and latest checkpoint as `last.pth`.

If you want to share a trained checkpoint, add it to a release or provide a download link and include instructions to place it under `checkpoints/`.

## Results

Include a short summary of expected results or current baseline here. Example:

- Baseline (ResNet50, standard aug, lr=1e-3): Val accuracy: ~85%
- With stronger augmentation and longer training: Val accuracy: ~88–90%

Replace the above with your actual results when available.

## Tips & troubleshooting

- If training is slow, reduce image size or batch size, or use a pretrained backbone.
- For class imbalance, consider weighted sampling or focal loss.
- Use learning rate schedulers (CosineAnnealing, ReduceLROnPlateau) to improve convergence.
- If GPU memory is limited, enable gradient accumulation or use mixed precision training (torch.cuda.amp).

## Contributing

Contributions are welcome. Suggested improvements:
- Add more models (EfficientNet, Vision Transformers)
- Add model explainability (Grad-CAM)
- Add hyperparameter tuning scripts

To contribute: fork the repo, create a feature branch, and open a pull request describing your changes.

## License

This project is distributed under the MIT License. See LICENSE for details.

## Contact

If you have questions or want to collaborate, open an issue or reach out to the repository owner: Ksh1t1zsharma.

---

Notes:
- Edit this README to add specific dataset links, exact commands, and real benchmark results once you run experiments.
