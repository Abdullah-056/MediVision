# MediVision 🏥

Convert handwritten medicine prescriptions into digital format using AI.

## What is MediVision?

MediVision uses deep learning to recognize and classify medicine names from handwritten prescription images. Perfect for healthcare digitization.

## ⚡ Quick Start

### 1. Download Required Files

- **Dataset**: [Doctors Handwritten Prescription Dataset](https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset)
- **Model Weights**: [ResNet18](https://www.kaggle.com/models/khadidjabrakta/resnet18/PyTorch/default/1)

### 2. Install Dependencies

```bash
pip install torch torchvision pandas scikit-learn pillow opencv-python matplotlib numpy
```

### 3. Organize Folders

```
your-project/
├── Dataset/
│   ├── Training/
│   ├── Validation/
│   └── Testing/
├── resnet18-f37072fd.pth
└── handwritten_latest.ipynb
```

### 4. Run the Notebook

Open `handwritten_latest.ipynb` in Jupyter and run all cells sequentially.

## 📊 What You Get

- **78 Medicine Classes** - Trained on 4,680 prescription images
- **320×320 Images** - Standardized preprocessing
- **92-95% Accuracy** - High precision predictions
- **Confidence Scores** - Know how sure the model is

## 🎯 Use the Model

```python
from medicine_predictor import MedicinePredictor

predictor = MedicinePredictor("./checkpoints_resnet18/best_resnet18.pt")
result = predictor.predict("prescription.png", top_k=5)

print(f"Medicine: {result['top_prediction']}")
print(f"Confidence: {result['confidence']*100:.2f}%")
```

## 📈 Dataset Breakdown

| Split | Images | Purpose |
|-------|--------|---------|
| Training | 3,120 | Train model |
| Validation | 780 | Tune during training |
| Testing | 780 | Evaluate final model |

## 🔧 Model Info

- **Architecture**: ResNet18 (Deep CNN)
- **Input**: 320×320 RGB images
- **Output**: Medicine name + confidence score
- **Training**: ~10 epochs with early stopping

## 📁 Output Files

After training, you'll get:
- `best_resnet18.pt` - Trained model
- `label_map.json` - Medicine name mapping
- `confusion_matrix_resnet18.csv` - Detailed accuracy report

## ❓ Troubleshooting

**Problem**: Images not found  
**Solution**: Check dataset folder paths match the code

**Problem**: Out of memory  
**Solution**: Reduce BATCH_SIZE from 32 to 16

**Problem**: Low accuracy  
**Solution**: Ensure images are 320×320 before training

## 📚 More Info

- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [PyTorch Docs](https://pytorch.org/)
- [Kaggle Dataset](https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset)

---

**Ready?** Start with the Jupyter notebook! 🚀
