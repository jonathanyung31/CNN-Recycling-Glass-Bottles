# Glass Bottle Defect Classification – Krones Vision AI Challenge

Binary classifier that decides whether an empty glass bottle is reusable (0) or not reusable (1), built for the THD × Krones Vision AI Challenge on Kaggle.

**Result:** F1 0.968 on the private leaderboard (0.967 public).

## Approach
- **Data:** 35,342 industrial bottle-base images with class imbalance. I cropped each image to the annotated bottle region (ROI) from the COCO annotations.
- **Augmentation:** Flips, 180° rotation, colour jitter, random grayscale and small shifts, chosen because bottle bases look the same from any angle.
- **Model:** ImageNet-pretrained EfficientNet-B3 at 512×512 with a single-output head.
- **Training:** Two phases. The head trained alone for 5 epochs, then the full network trained with AdamW and OneCycleLR in mixed precision.
- **Imbalance:** Weighted BCE loss and label smoothing. I selected the best checkpoint by validation auROC on a stratified 10% hold-out.
- **Deployment:** Exported to ONNX and run with ONNX Runtime. The decision threshold is 0.55.

## Files
- Training notebook: data loading, model, training loop, ONNX export
- Evaluation notebook: ONNX Runtime inference pipeline and submission file
- `presentation.pdf`: project presentation
- `requirements.txt`: Python dependencies

## Data
The images belong to the competition and are not included. See the Kaggle competition page for access.
