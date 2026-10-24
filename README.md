# ENGE707 — Cat and Dog Facial Feature Classification

This project uses an ImageNet-pretrained **ResNet18** model to classify eight cat and dog facial features. The notebooks cover data exploration, preprocessing, training, fine-tuning, evaluation and Kaggle prediction generation.

This README explains how to **set up and run the project**. The report and notebook outputs contain the experimental results and analysis.

## 1. Requirements

Use Python 3.12 with Jupyter Notebook or the Jupyter extension in VS Code. Install the required packages:

```bash
python -m pip install torch torchvision numpy pandas matplotlib pillow scikit-learn ipykernel jupyter
```

## 2. Set up the dataset

Download the images and CSV files from the [ENGE707 Kaggle competition](https://www.kaggle.com/competitions/enge-707-2026-s-2-cat-and-dog-facial-feature-classification-ch/overview).

For **local execution**, arrange the files as follows:

```text
project/
├── train/
│   └── train/                   # Training images (.jpg)
├── test/
│   └── test/                    # Kaggle test images (.jpg)
├── train_labels.csv
├── sample_submission.csv
├── data_pipeline.py
├── Task1_Data_Exploration.ipynb
├── Task2_Data_Preprocessing.ipynb
├── Task3_Model_Architecture.ipynb
├── Task4_Training_Performance_Analysis.ipynb
├── Task4_ResNet18_FineTuning.ipynb
├── Task5_Model_Evaluation.ipynb
├── best_resnet18.pth
├── best_resnet18_finetuned.pth
├── training_history.json
└── finetuning_history.json
```

The original image directories are **not included** in the submitted project files. If either competition CSV is missing, download it from Kaggle as well. Open the **project folder** in VS Code or Jupyter so the notebooks run with this folder as their current working directory.

`data_pipeline.py` provides the shared image preprocessing, class encoding and DataLoaders. Its `create_dataloaders()` function requires **both image folders and both CSV files**, even when only the validation set is being evaluated.

## 3. Notebook guide

| Notebook | Purpose |
| --- | --- |
| `Task1_Data_Exploration.ipynb` | Inspect the dataset and its classes |
| `Task2_Data_Preprocessing.ipynb` | Demonstrate preprocessing, splitting and augmentation |
| `Task3_Model_Architecture.ipynb` | Define/train the baseline model and generate `submission.csv` |
| `Task4_Training_Performance_Analysis.ipynb` | Plot the saved baseline training history |
| `Task4_ResNet18_FineTuning.ipynb` | Fine-tune the model and generate `submission_finetuned.csv` |
| `Task5_Model_Evaluation.ipynb` | Evaluate the saved fine-tuned checkpoint on validation data |

For the full workflow, follow the order shown in the table. Run **each notebook's cells in order**. Tasks 3, 4 (fine-tuning) and 5 load data independently through the shared pipeline, so they do not need variables left over from earlier notebooks.

## 4. Reproduce the evaluation and submissions

**Review the saved work without retraining:**

- Open Tasks 1 and 2 to view their saved outputs. Re-running their dataset checks is optional.
- Run `Task4_Training_Performance_Analysis.ipynb` with `training_history.json` in the project folder.
- In `Task4_ResNet18_FineTuning.ipynb`, leave `RUN_FULL_FINETUNING = False` and `DOWNLOAD_COMPETITION_DATA = False`. Run cells from the beginning **up to (but not including) "Generate Kaggle Predictions"** to load the checkpoints and saved fine-tuning history. This review does not need the image dataset.

**Re-evaluate the final model:**

- With the **complete dataset** and `best_resnet18_finetuned.pth` in the project folder, run all cells in `Task5_Model_Evaluation.ipynb` in order.
- The notebook loads the saved checkpoint and produces validation metrics and a confusion matrix. It **does not retrain** the model or create a Kaggle submission.

**Regenerate Kaggle prediction CSVs:**

- **Baseline:** In `Task3_Model_Architecture.ipynb`, leave `RUN_FULL_TRAINING = False` and run its setup and checkpoint-loading cells, followed by the test-prediction and submission sections. It uses `best_resnet18.pth` to write `submission.csv`.
- **Fine-tuned:** In `Task4_ResNet18_FineTuning.ipynb`, leave `RUN_FULL_FINETUNING = False` and `DOWNLOAD_COMPETITION_DATA = False`, then run **all cells in order**, including the prediction section. It uses `best_resnet18_finetuned.pth` to write `submission_finetuned.csv`.

Both prediction workflows require the **complete Kaggle dataset** and may overwrite existing submission CSVs. The final competition submission is `submission_finetuned.csv`, in the required `image_id,class` format.

## 5. Reproduce training (optional)

Training is disabled by default so the saved models can be used without repeating it.

- **Baseline training:** In Task 3, change `RUN_FULL_TRAINING` to `True` and run the training section. New files are saved to `training_outputs/best_resnet18_new_run.pth` and `training_outputs/training_history_new_run.json`.
- **Fine-tuning:** In Task 4's fine-tuning notebook, supply `best_resnet18.pth`, set `RUN_FULL_FINETUNING = True` and run through the training section. New files are saved as `best_resnet18_finetuned_reproduced.pth` and `finetuning_history_reproduced.json`.

These are **separate filenames** from the included checkpoints. By default, the evaluation and submission notebooks still use the original saved checkpoints; to use a newly trained model, update the checkpoint path explicitly. A fresh run can produce different results because package versions and all sources of training randomness were not fully fixed.

## 6. Running on Kaggle

Locally, use the folder structure in Section 2. On Kaggle, Tasks 3 and 4 can use the dataset extracted to `/kaggle/working/competition_data`. Make `data_pipeline.py` available in the working directory or at the pipeline mount already supported in the notebooks. If the dataset or module is attached at another location, adjust Task 3's `DATA_DIR` or Task 4's path resolver accordingly.

Task 4 can optionally download the dataset by setting `DOWNLOAD_COMPETITION_DATA = True`; this requires competition access and the Kaggle CLI. **Task 5 expects the complete dataset, pipeline and fine-tuned checkpoint together in its current working directory**, so the local layout is the simplest way to re-run validation.
