# ENGE707 — Cat and Dog Facial Feature Classification

This project uses **ImageNet-pretrained ResNet18** to classify images into eight cat and dog facial-feature classes: `CatEye`, `CatFace`, `CatMouth`, `CatNose`, `DogEye`, `DogFace`, `DogMouth`, and `DogNose`.

The workflow covers data exploration, preprocessing, baseline training, partial fine-tuning, validation evaluation, and Kaggle submissions. The recorded final fine-tuned model achieved **98.76% validation accuracy** and **94.14% accuracy on Kaggle's public leaderboard test subset**.

## Requirements

- Python 3.12, Jupyter Notebook or VS Code with the Python/Jupyter extensions. Python 3.12 was used for part of the project; the experiments were not all performed in a single pinned environment.
- Install the notebook dependencies:

```bash
python -m pip install torch torchvision numpy pandas matplotlib pillow scikit-learn ipykernel jupyter
```

- Download the competition dataset from the [ENGE707 Cat and Dog Facial Feature Classification competition](https://www.kaggle.com/competitions/enge-707-2026-s-2-cat-and-dog-facial-feature-classification-ch/overview). Dataset images and competition CSVs are not bundled in the project repository.
- Keep **`data_pipeline.py`** in the project root; it provides the shared transforms, class mapping and DataLoaders.
- Dependency versions are not pinned in a lock file, so the original software environment is not reproduced exactly. The data split uses `random_state=42`, but this does not fully seed training and augmentation randomness.

## Project files

| File | Purpose |
| --- | --- |
| `Task1_Data_Exploration.ipynb` | Explore class counts, image properties and duplicate images |
| `Task2_Data_Preprocessing.ipynb` | Deduplicate training records, create the stratified split and demonstrate transforms/DataLoaders |
| `Task3_Model_Architecture.ipynb` | Define and optionally train baseline ResNet18; generate baseline Kaggle predictions |
| `Task4_Training_Performance_Analysis.ipynb` | Display the initial training/validation loss and accuracy curves |
| `Task4_ResNet18_FineTuning.ipynb` | Configure partial fine-tuning, inspect saved histories/checkpoints and generate fine-tuned Kaggle predictions in the project notebook |
| `Task5_Model_Evaluation.ipynb` | Re-evaluate the best fine-tuned checkpoint on validation images; calculate loss, accuracy, precision, recall, F1-score and confusion matrix |
| `data_pipeline.py` | Reusable dataset, transforms, labels and DataLoader implementation |

The included Task 5 notebook has eight cells, with validation inference, a classification report, a confusion matrix and saved evaluation outputs.

## Dataset and preprocessing

- Original labelled training images: **63,660**; unlabelled Kaggle test images: **29,443**.
- Training data contains **15 duplicate pairs** identified using SHA-256. One record from each pair is excluded, leaving **63,645** labelled examples. The original image files are **not** deleted.
- Stratified 80/20 training-validation split (`random_state=42`): **50,916 training** and **12,729 validation** examples.
- All loaders use **batch size 32**; only the training loader shuffles, and incomplete last batches are retained.
- Convert to RGB, resize while preserving the aspect ratio and apply centered black padding to **224 × 224**. The resize uses bilinear interpolation with antialiasing.
- Normalise using ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- Training augmentation uses random horizontal flipping (`p=0.5`), rotation up to ±15° and brightness variation (factor 0.8–1.2). Validation and test data use deterministic transforms without random augmentation.

For local use, the expected layout is:

```text
project/
├── train_labels.csv                   # download from Kaggle
├── sample_submission.csv              # download from Kaggle
├── train/train/                        # training images
├── test/test/                          # unlabelled test images
├── data_pipeline.py
├── Task1_Data_Exploration.ipynb
├── Task2_Data_Preprocessing.ipynb
├── Task3_Model_Architecture.ipynb
├── Task4_Training_Performance_Analysis.ipynb
├── Task4_ResNet18_FineTuning.ipynb
├── Task5_Model_Evaluation.ipynb        # complete evaluation version
├── best_resnet18.pth                   # recorded baseline checkpoint
├── training_history.json              # recorded baseline history
├── best_resnet18_finetuned.pth         # recorded fine-tuned checkpoint
├── finetuning_history.json             # recorded fine-tuning history
├── submission.csv                      # baseline predictions, if generated
└── submission_finetuned.csv            # fine-tuned predictions, if generated
```

The notebooks, pipeline, recorded checkpoints and histories are included. `.gitignore` excludes the image directories and all CSV files; obtain the competition CSVs separately and generate submissions when needed. Open VS Code in the project root, select the intended Python kernel, and ensure `Path.cwd()` points to that directory.

## Model and training configuration

### Initial ResNet18

- Load ImageNet-pretrained ResNet18 and replace the original 1,000-class fully connected layer with `nn.Linear(512, 8)`.
- Freeze the pretrained feature extractor and train only `fc` for **10 epochs**, using cross-entropy loss and Adam with learning rate **0.001**.
- Keep BatchNorm running statistics fixed during training.
- The **recorded best checkpoint** is `best_resnet18.pth`, selected at baseline epoch **10** by validation accuracy.

### Partial fine-tuning

- Start from the recorded baseline checkpoint, then unfreeze `layer4` and `fc`; keep the earlier convolutional groups frozen.
- Fine-tune for **5 additional epochs** with separate Adam learning rates: **0.00001** for `layer4` and **0.0001** for `fc`.
- Keep BatchNorm running statistics fixed. Trainable BatchNorm affine parameters within `layer4` can still update.
- The **recorded best checkpoint** is `best_resnet18_finetuned.pth`, selected at fine-tuning epoch **4** by validation accuracy.

The recorded artifact names above are distinct from the filenames created by optional *new* training runs.

## Running the notebooks

For a complete project walkthrough, follow Tasks 1 → 2 → 3 → Task 4 Training Performance Analysis → Task 4 ResNet18 Fine-Tuning → Task 5. Execute cells in order within each notebook using the selected kernel. Tasks 3, 4 fine-tuning and 5 construct their own DataLoaders through `data_pipeline.py`; they do not depend on variables from Tasks 1 or 2. Saved-artifact review can be run directly as described below.

### Review saved results without retraining

1. Open Tasks 1 and 2 to inspect their saved outputs. Re-running Task 1's full checks hashes and decodes the image datasets; this is unnecessary for reviewing existing findings.
2. Run `Task4_Training_Performance_Analysis.ipynb` with `training_history.json` in the working directory to display the recorded baseline curves.
3. In `Task4_ResNet18_FineTuning.ipynb`, keep `RUN_FULL_FINETUNING = False` and `DOWNLOAD_COMPETITION_DATA = False`. Provide both recorded checkpoints and `finetuning_history.json` in the project root, then execute all cells in order and stop before **Generate Kaggle Predictions**. Include the **Load Dataset** and optional fine-tuning cells: they define the shared discovery helper and output-path variables, even though training and dataset loading are skipped. This review does not require competition data. Continuing into prediction loads the data, performs test inference and writes a CSV.

### Re-evaluate the saved fine-tuned model

Run all eight cells of `Task5_Model_Evaluation.ipynb` in order from the documented local project root. Provide `data_pipeline.py`, `best_resnet18_finetuned.pth` and the **complete competition dataset**: `train_labels.csv`, `train/train/`, `sample_submission.csv` and `test/test/`. The shared pipeline constructs all three DataLoaders and hashes training files, so test files are required even for validation-only evaluation.

Task 5 evaluates 12,729 validation images using the saved checkpoint and computes accuracy, cross-entropy loss, class precision/recall/F1 and a confusion matrix. It performs inference without training or generating submission files. It imports the pipeline directly and uses `PROJECT_DIR = Path.cwd()` for data and the checkpoint; it does not use Task 4's discovery helper.

### Reproduce training (optional)

Use the complete competition dataset and run each notebook from its setup cells in a fresh kernel:

- **Baseline:** In Task 3, set `RUN_FULL_TRAINING = True` in **Full Training (optional)** and execute through that section. It trains for 10 epochs and saves `training_outputs/best_resnet18_new_run.pth` and `training_outputs/training_history_new_run.json`. The output root is the project directory locally, or `/kaggle/working` on Kaggle. Existing new-run files can be overwritten. Task 3 initializes ImageNet weights (downloading them if uncached) and runs a one-batch training smoke test on a model copy before full training.
- **Fine-tuning:** Provide `best_resnet18.pth`, set `RUN_FULL_FINETUNING = True`, and run through the training section. The notebook trains `layer4` and `fc` for five additional epochs. It saves `best_resnet18_finetuned_reproduced.pth` and `finetuning_history_reproduced.json` in `PROJECT_DIR` locally or `/kaggle/working` on Kaggle. It refuses to overwrite existing reproduced files.
- **Selecting new artifacts:** The baseline checkpoint-loading section still loads `best_resnet18.pth`; the performance-analysis notebook reads `training_history.json`. The fine-tuning comparison/history sections prefer original artifacts over reproduced files. To evaluate, plot or predict with a new run, explicitly select its checkpoint/history paths. Fine-tuning a newly reproduced baseline also requires selecting that baseline in the checkpoint-loading section.

Both training loops save a checkpoint only when validation accuracy improves. Stop before submission sections when only training is intended. New training runs may differ because dependency versions and training/augmentation randomness are not fully fixed.

### Local and Kaggle paths

- **Local:** Start from the project root shown above, with `data_pipeline.py` and the recorded artifacts beside the notebooks. The documented layout requires no dataset-path edits.
- **Baseline notebook:** `DATA_DIR` selects `/kaggle/working/competition_data` when its `train_labels.csv` exists; otherwise it uses the working directory. Setup adds `/kaggle/input/datasets/shawnleeyz/pipeline` to the import path if present. The submission section reads `DATA_DIR / "sample_submission.csv"`.
- **Fine-tuning notebook:** `resolve_pipeline_and_data_dir()` searches for the pipeline in `PROJECT_DIR` first, then `/kaggle/input/datasets/shawnleeyz/pipeline/data_pipeline.py`, and adds its directory to the import path. Dataset candidates, in priority order, are `/kaggle/working/competition_data`, `PROJECT_DIR`, and `PROJECT_DIR / "competition_data"`. Training and prediction call this same helper. Prediction resolves paths even with `RUN_FULL_FINETUNING = False`, retains the selected `data_dir`, and reads `data_dir / "sample_submission.csv"`.
- **Kaggle download:** In the fine-tuning notebook, `DOWNLOAD_COMPETITION_DATA = True` downloads/extracts into `/kaggle/working/competition_data`, independently of the training flag. This requires the Kaggle CLI, authenticated competition access and acceptance of the competition rules. To install the optional CLI, use `python -m pip install kaggle`. The supported downloaded-data and pipeline-mount locations above are handled automatically; no submission-cell path corrections are needed.
- **Other mounts and artifacts:** Arbitrary competition-data mounts are not automatically searched. For those layouts, configure Task 3's `DATA_DIR` or Task 4's helper candidates, and the pipeline location if its mount differs. Keep the original checkpoints and histories in the working directory for the simplest Kaggle setup. Task 3 and Task 5 load their checkpoints there; Task 4 also supports the baseline mount `/kaggle/input/datasets/shawnleeyz/best-resnet18/best_resnet18.pth`. Task 5 requires its complete project-root layout even on Kaggle, or explicit configuration of its direct import/data/checkpoint paths. Use a writable working directory for generated outputs.

These instructions have been compared with the source code using static inspection; the workflows were not rerun for this documentation update.

## Expected results (for verification)

| Model | Best validation accuracy | Validation cross-entropy loss | Kaggle public leaderboard accuracy | Saved epoch |
| --- | ---: | ---: | ---: | --- |
| Initial ResNet18 | 95.98% | 0.1189 | 90.60% | Baseline 10/10 |
| Fine-tuned ResNet18 | **98.76%** | **0.0383** | **94.14%** | Fine-tuning 4/5 |

The fine-tuned model's evaluation on **12,729 validation images** also recorded **98.98% macro precision**, **98.69% macro recall**, and **98.83% macro F1-score**. Use these values to compare the notebook outputs with the original recorded run.

These are reference values from the recorded experiments. Fresh training runs may differ, and Kaggle scores are from the original submissions.

## Kaggle predictions and submission

| Prediction file | Intended checkpoint | Recorded public leaderboard accuracy |
| --- | --- | ---: |
| `submission.csv` | `best_resnet18.pth` | 90.60% |
| `submission_finetuned.csv` | `best_resnet18_finetuned.pth` | 94.14% |

For inference, use deterministic test preprocessing, set `model.eval()`, run within `torch.no_grad()`, select each image's highest-scoring output with `argmax` and map the predicted class index back to its class name.

Both inspected submission CSV files contained **29,443 predictions** and the exact column headings and image-ID order from `sample_submission.csv`:

```csv
image_id,class
```

- **Baseline CSV:** In Task 3, leave `RUN_FULL_TRAINING = False`, execute setup through **Load the Saved Checkpoint**, then run **Verify the Test DataLoader**, **Generate Test Predictions** and **Create the Kaggle Submission CSV**. Skip **One-Batch Smoke Test** if no training at all is intended; it only tests a model copy and is not needed for inference. The checkpoint is `best_resnet18.pth`; the template is read from `DATA_DIR`; output is `submission.csv` in the working directory.
- **Fine-tuned CSV:** Keep `RUN_FULL_FINETUNING = False`, execute all preceding cells in order through **Compare Baseline and Fine-Tuned Models**, then run **Generate Kaggle Predictions**. The default recorded checkpoint is `best_resnet18_finetuned.pth`. The shared helper resolves the pipeline and dataset without enabling training. The template comes from `data_dir`; output is `PROJECT_DIR / "submission_finetuned.csv"`.

Both workflows require the complete competition dataset through the shared pipeline and may overwrite their existing submission CSV. Both use the template's image-ID order and class-name mapping. Upload the generated CSV to the competition submission page. Kaggle leaderboard scores are recorded submission results, not metrics computed by local inference.
