# ENGE707 — Cat and Dog Facial Feature Classification

This project classifies images into eight classes using a pretrained ResNet18 model: `CatEye`, `CatFace`, `CatMouth`, `CatNose`, `DogEye`, `DogFace`, `DogMouth`, and `DogNose`.

## Requirements

- Python 3.12 (used for the project) and Jupyter / VS Code.
- Install the Python packages used by the notebooks: `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `Pillow`, `scikit-learn`, `ipykernel`, and `jupyter`.
- For example: `python -m pip install torch torchvision numpy pandas matplotlib pillow scikit-learn ipykernel jupyter`.
- The Kaggle competition dataset must be downloaded separately; it is not included in this package.
- The shared `data_pipeline.py` module, created from Task 2 preprocessing, is **required for Tasks 3 and 4 fine-tuning**. Place your existing copy beside the notebooks. It was not part of the five supplied notebook files.

## Files and execution order

| File | Purpose |
| --- | --- |
| `Task1_Data_Exploration.ipynb` | Dataset statistics, inspection and duplicate checks |
| `Task2_Data_Preprocessing.ipynb` | Deduplication, stratified 80/20 split, image transforms and DataLoaders |
| `Task3_Model_Architecture.ipynb` | Baseline ResNet18 architecture, optional 10-epoch training, checkpoint loading and Kaggle test predictions |
| `Task4_Training_Performance_Analysis.ipynb` | Original 10-epoch accuracy/loss curves |
| `Task4_ResNet18_FineTuning.ipynb` | Partial fine-tuning of `layer4` and `fc`, saved curves and checkpoint comparison |
| `data_pipeline.py` | Reusable DataLoaders for Tasks 3 and 4; copy this from your existing project |

Open VS Code in the project root and select the project Python environment as the notebook kernel. Keep the current working directory at the project root (`Path.cwd()` should point there).

## Dataset and saved model files

Place the extracted competition files in the project root for local runs:

```text
project/
├── train_labels.csv
├── sample_submission.csv
├── train/train/                  # training JPEGs
├── test/test/                    # test JPEGs
├── data_pipeline.py             # from your existing project
├── best_resnet18.pth            # original checkpoint, if available
├── training_history.json       # original training metrics, if available
├── best_resnet18_finetuned.pth  # fine-tuned checkpoint, if available
├── finetuning_history.json      # fine-tuning metrics, if available
└── *.ipynb
```

### Inspect saved results without retraining

1. Run `Task4_Training_Performance_Analysis.ipynb` with `training_history.json` to display the original learning curves.
2. Run `Task4_ResNet18_FineTuning.ipynb` with both saved checkpoints and `finetuning_history.json`. Leave `RUN_FULL_FINETUNING = False`. This shows the saved metrics and compares checkpoints; it does **not** train again.
3. In Task 3, the original baseline checkpoint `best_resnet18.pth` must be present before generating test predictions. Inference generates `submission.csv` following `sample_submission.csv` order and class names.

### Reproduce training (optional)

- **Baseline:** Task 3 uses ImageNet pretrained ResNet18 with only `fc` trainable, Adam learning rate `0.001`, batch size 32 and 10 epochs. Set `RUN_FULL_TRAINING = True` only when intentionally retraining. New files are saved in `training_outputs/` to avoid overwriting the downloaded baseline checkpoint.
- **Fine-tuning:** Task 4 fine-tuning starts from `best_resnet18.pth` and updates `layer4` (Adam learning rate `1e-5`) and `fc` (`1e-4`) for five additional epochs, retaining frozen BatchNorm running statistics. Set `RUN_FULL_FINETUNING = True` and supply the same training data and `data_pipeline.py`. Reproduced outputs use filenames ending in `_reproduced` to avoid overwriting the originals.
- **Kaggle GPU:** In the Task 4 fine-tuning notebook, `DOWNLOAD_COMPETITION_DATA = True` can download and extract the dataset if running on Kaggle with competition access and its CLI available. The notebook also looks for Kaggle datasets containing the baseline checkpoint and pipeline module. Use GPU acceleration for full training.
- Augmentation includes random horizontal flips, rotation, and brightness changes on training images only. The random operations mean repeated runs may have slightly different results.

## Recorded results

These are the reported results from the saved experiments, **not freshly recomputed by this README**:

| Experiment | Best validation accuracy | Saved epoch |
| --- | ---: | ---: |
| Original ResNet18 | 95.98% | 10 of 10 |
| Partial fine-tuning | 98.76% | 4 of 5 additional epochs |

The baseline Kaggle leaderboard accuracy reported was **90.60%**. A Kaggle score for the fine-tuned checkpoint has not yet been established here; do not assume it equals the validation accuracy.

## Submission notes

Keep `sample_submission.csv` column names and image IDs unchanged and use the **exact class names** from the training dataset. Submit the prediction CSV on Kaggle. Keep the leaderboard evidence separate from the saved validation metrics. Dataset images, `.pth` files and `training_history.json`/`finetuning_history.json` are not included in the download package below; use your own existing copies.
