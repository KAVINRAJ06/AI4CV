# SegAL-VLM-X: Text-Guided Explainable Active Learning for Remote Sensing Semantic Segmentation

## 1. Overview
**SegAL-VLM-X** is an advanced research framework designed for semantic segmentation on the **LoveDA** remote sensing dataset. It leverages Vision-Language Models (VLMs) and Explainable Active Learning (XAL) to achieve high performance with efficient data labeling.

### Key Features
- **Vision-Language Fusion**: Integrates **ViT/DINOv2** (Vision) and **CLIP** (Text) via Cross-Attention.
- **Text-Guided Segmentation**: Uses class-specific text prompts (e.g., "satellite image of a building") to guide the segmentation decoder.
- **Explainable Active Learning (XAL)**: Introduces **MEM (Multi-modal Error Mask)**, a novel query strategy combining model uncertainty (Entropy) and text-visual relevance (Attention Maps).
- **Iterative AL Loop**: Fully automated loop: Train $\rightarrow$ Predict $\rightarrow$ Rank (MEM) $\rightarrow$ Query Oracle $\rightarrow$ Update Pool $\rightarrow$ Retrain.
- **Robust Data Handling**: Native support for LoveDA's RGB masks with vectorized conversion to class indices.

---

## 2. System Architecture

The framework follows a modular 6-layer structure:

### Layer 1: Data Management (`dataset/`)
- **LoveDA Dataset**: Handles loading, splitting (Train/Val/Test), and transforms.
- **Mask Handling**: Automatically converts **RGB masks** (e.g., Building=Red) to **Class Indices** (0-5).
  - *Classes*: Building (0), Road (1), Water (2), Barren (3), Forest (4), Agricultural (5).
  - *Ignore*: Background (255).
- **Prompts**: Manages text prompts for each class (defined in `dataset/prompts.json`).

### Layer 2: Model Architecture (`models/`)
- **Vision Encoder**: `timm` based ViT (e.g., `vit_base_patch16_224`) for feature extraction.
- **Text Encoder**: Frozen **CLIP** text encoder for embedding class prompts.
- **Cross-Attention**: Fuses visual features (Query) with text embeddings (Key/Value) to inject semantic guidance.
- **Decoder**: CNN-based decoder with residual connections and bilinear upsampling for dense prediction.

### Layer 3: Explainable AI (XAI) (`xai/`)
- **Entropy**: Quantifies pixel-wise model uncertainty.
- **Attention Maps**: Extracts cross-attention weights to visualize model focus.
- **MEM (Multi-modal Error Mask)**:
  $$ \text{MEM} = \alpha \cdot \text{TextAttention} + (1-\alpha) \cdot \text{Entropy} $$
  High MEM scores indicate regions where the model is uncertain *and* the text guidance is ambiguous, prioritizing these for labeling.

### Layer 4: Active Learning (`active_learning/`)
- **Oracle**: Simulates human annotator using ground truth masks.
- **Sampler**: Ranks unlabeled samples based on MEM scores.
- **Dice Selector**: Optional diversity-based selection.

### Layer 5: Training & Control (`training/`)
- **Controller**: Manages the AL loop (Training $\leftrightarrow$ Selection).
- **Loss**: Cross-Entropy Loss with `ignore_index=255`.
- **Metrics**: mIoU, Pixel Accuracy, Class-wise IoU.
- **Checkpointing**: Saves `last_epoch.pth` and `best_miou.pth`.

### Layer 6: Visualization (`visualization/`)
- **Tools**: Generates plots for Input, Ground Truth, Prediction, and MEM Heatmaps.

---

## 3. Directory Structure
```
SegAL_VLM_LoveDA/
├── active_learning/    # Oracle, Sampler, AL Logic
├── configs/            # YAML Configs (Model, Training, AL)
├── data/               # Dataset Directory
│   └── LoveDA/         # Train/Val/Test Splits
├── dataset/            # Data Loaders & Mask Conversion
├── experiments/        # Logs, Checkpoints, Plots
│   ├── checkpoints/    # Saved Models (.pth)
│   └── logs/           # Metrics & Visualizations
├── models/             # Neural Network Modules
├── training/           # Train/Val Loops
├── visualization/      # Plotting Scripts
├── xai/                # MEM & Explainability
├── main_train.py       # Entry point for AL Training Loop
├── main_eval.py        # Entry point for Evaluation
├── main_visualize.py   # Entry point for Visualization
└── requirements.txt    # Python Dependencies
```

---

## 4. Usage

### Prerequisites
- Python 3.8+
- PyTorch, Torchvision
- Transformers, Timm, Matplotlib, Pandas

Install dependencies:
```bash
pip install -r requirements.txt
```

### 1. Training (Active Learning Loop)
Starts the iterative active learning process.
```bash
python main_train.py
```
*Config*: `configs/active_learning.yaml` (Controls rounds, budget, pool size).

### 2. Evaluation
Evaluates the best model on the validation set.
```bash
python main_eval.py --checkpoint experiments/checkpoints/best_miou.pth
```
*Output*: Console report with Overall mIoU and Class-wise IoU.

### 3. Visualization
Visualizes model predictions and MEM heatmaps for specific samples.
```bash
# Visualize specific index
python main_visualize.py --index 10

# Visualize specific checkpoint
python main_visualize.py --checkpoint experiments/checkpoints/best_miou.pth --index 5
```
*Output*: `visualization_result_{index}.png`

---

## 5. Configuration

- **`configs/model.yaml`**: Model hyperparameters (Encoder types, hidden dimensions, dropout).
- **`configs/training.yaml`**: Training settings (LR, Batch Size, Epochs, Optimizer).
- **`configs/active_learning.yaml`**: AL strategy (Initial ratio, Budget per round, Total rounds).

---

## 6. Novelty & Contribution
**SegAL-VLM-X** addresses the "black-box" nature of Deep Active Learning by introducing **Explainability**. Instead of relying solely on uncertainty (which can be noisy), it uses **Text-Visual Consistency** (via Cross-Attention) to validate uncertainty. If the model is uncertain *despite* strong text guidance, it indicates a genuine hard example worthy of human labeling.
