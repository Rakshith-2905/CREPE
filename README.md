# CREPE: CLIP Representation Enhanced Predicate Estimation

![](https://img.shields.io/badge/pytorch-green)

This repository hosts the official PyTorch implementation for: "CREPE: Learnable Prompting With CLIP Improves Visual Relationship Prediction"

## Abstract
In this project, we explore the potential of Vision-Language Models (VLMs), specifically CLIP, in predicting visual object relationships, which involves interpreting visual features from images into language-based relations. We adopt the UVTransE relation prediction framework, which learns the relation as a translational embedding with subject, object, and union box embeddings from a scene. We propose CREPE (CLIP Representation Enhanced Predicate Estimation), a novel approach that utilizes text-based representations for all three bounding boxes and introduces a novel contrastive training strategy to automatically infer the text prompt for union-box. Our approach achieves state-of-the-art performance in predicate estimation, mR@5 27.79, and mR@20 31.95 on the Visual Genome benchmark, achieving a 15.3% gain in performance over recent state-of-the-art at mR@20.

## Installation
The project requires Python 3.8 or later and makes use of PyTorch for training the models.

1. Clone the repository:
```bash
git clone https://github.com/LLNL/CREPE
cd CREPE
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Dataset
We use the Visual Genome benchmark dataset for this project. You can download it from the [official website](http://visualgenome.org/). After downloading, place the dataset in the `./datasets` directory.

## Training the Models

### Training the Prompter
The `train_prompter.py` script is used to train the prompter model. You can start the training process using the following command:

```bash
python train_prompter.py --n_contex_vectors 8 --token_position 'middle' --num_predicates 50 \
        --epochs 500 --learning_rate 2e-3 --batch_size 256 \
        --use_cuda True --out_dir './output' --data_dir 'datasets/pred_dicts_train_cmr'
```

You can also specify other command line arguments as per your requirements.

### Training the Classifier
After training the prompter, the obtained features are used for training the classifier. You can train the classifier using the `train_classifier.py` script as follows:

```bash

python legacy_train_classifier.py --batch_size=1 --learning_rate 0.001 --which_epoch=500 --train_epochs 100 --save_freq 1 --use_cuda True \
        --n_context_vectors=8 --token_position middle --learnable_UVTransE True --update_UVTransE True --is_non_linear True --num_predicates=50 \
        --checkpoints_dir_prompt=output/2023-05-09_19-18-07/checkpoints --out_dir=output/2023-05-09_19-18-07 \
        --data_dir='datasets/VG/np_files'
```

Just like the prompter, you can specify other command line arguments as per your requirements.

## Acknowledgements

We thanks the authors of the UVTransE and CLIP papers for their inspiring and foundational work. Our sincere thanks also goes to the authors of the [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) repository helped us in deriving the FRCNN features and metrics implementation.

## Contact
For any queries, feel free to reach out at `rakshith.2905@gmail.com`.


---
