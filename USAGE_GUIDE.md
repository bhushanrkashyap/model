# Waste Classifier - Usage Guide

## 🎯 Quick Start

Your AI waste classification model is **trained and ready to use**! The model achieved **92.68% accuracy** on the test set.

---

## 📋 How to Test the Model

### 1. **Activate Virtual Environment** (Always do this first!)
```bash
cd /Users/bhushanrkaashyap/Desktop/ai/archive/DATASET
source venv/bin/activate
```

### 2. **Test on Random Images**
Run the simple test to see predictions on 10 random test images:
```bash
python simple_test.py
```

This will show:
- Predicted vs. actual labels
- Confidence scores
- Overall accuracy

**Example output:**
```
1. O_13387.jpg
   True: O | Predicted: O (100.0%) ✅

2. R_10820.jpg
   True: R | Predicted: R (55.3%) ✅
...
Accuracy: 8/10 (80.0%)
```

---

### 3. **Predict Single Images**
Classify any image from your dataset (or your own images!):
```bash
python predict_image.py DATASET/TEST/O/O_13387.jpg
```

**Example output:**
```
==================================================
PREDICTION RESULTS
==================================================

Organic/Wet Waste:
  █████████████████████████████████████████████████ 100.00%

Recyclable/Dry Waste:
   0.00%

==================================================
✅ CLASSIFICATION: 🍂 Organic/Wet Waste
   Confidence: 100.0%
==================================================
```

---

## 🗂️ Understanding the Classes

| Code | Type | Examples |
|------|------|----------|
| **O** | Organic/Wet Waste | Food scraps, garden waste, biodegradable materials |
| **R** | Recyclable/Dry Waste | Plastic bottles, paper, cardboard, metal cans |

---

## 📁 Project Files

### **Model Files**
- `waste_classifier_best.h5` - Trained model (92.68% accuracy, 23 MB)
- `class_names.json` - Class mappings ["O", "R"]

### **Scripts**
- `simple_test.py` - Test on random images from dataset
- `predict_image.py` - Predict single image with visualization
- `train_model.py` - Original training script (already completed)

### **Documentation**
- `README.md` - Setup instructions
- `TRAINING_COMPLETE.md` - Training results and metrics
- `PROJECT_DOCUMENTATION.md` - Comprehensive documentation
- `USAGE_GUIDE.md` - This file!

---

## 🖼️ Using Your Own Images

You can test the model on your own waste images:

1. Take a photo of waste (or find an image online)
2. Save it somewhere (e.g., `my_waste.jpg`)
3. Run prediction:
   ```bash
   python predict_image.py /path/to/my_waste.jpg
   ```

**Tips for best results:**
- Images should be clear and well-lit
- Focus on the waste item (not too much background)
- Works best with single items or similar waste types
- 224x224 pixels is ideal, but any size works (auto-resized)

---

## 📊 Model Performance

From the last training run:

- **Test Accuracy:** 92.68%
- **Test Loss:** 0.1896
- **Training Time:** ~15 minutes (10 epochs)
- **Architecture:** MobileNetV2 (transfer learning)
- **Parameters:** ~2.3 million

### Performance by Class
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| O (Organic) | ~93% | ~94% | ~93% |
| R (Recyclable) | ~92% | ~91% | ~92% |

---

## 🎨 Advanced Usage

### Batch Predictions
Create a script to process multiple images:
```python
import os
from predict_image import predict

images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
for img in images:
    predict(img)
```

### Integration Ideas
- **Web App:** Deploy with Flask/FastAPI
- **Mobile App:** Convert to TensorFlow Lite
- **API Service:** Create REST endpoint for predictions
- **Desktop App:** Build GUI with tkinter/PyQt

---

## 🔧 Troubleshooting

### "Module not found" errors
Make sure virtual environment is activated:
```bash
source venv/bin/activate
```

### "Image not found" errors
Check the path to your image:
```bash
ls -la DATASET/TEST/O/  # List available images
```

### Model loading issues
The scripts recreate the model architecture and load weights, avoiding serialization issues with the .h5 file.

---

## 🚀 Next Steps

Want to improve the model? Try:

1. **Add More Classes:** Train on metallic waste, hazardous waste, etc.
2. **Increase Accuracy:** 
   - Train longer (20-30 epochs)
   - Add more data augmentation
   - Try different architectures (EfficientNet, ResNet)
3. **Deploy It:**
   - Create a web interface
   - Make a mobile app
   - Set up an API service
4. **Real-world Testing:**
   - Test with actual waste photos
   - Collect feedback and retrain

---

## 📝 Summary

You now have:
- ✅ A trained AI model (92.68% accuracy)
- ✅ Scripts to test and use the model
- ✅ Complete documentation
- ✅ Ready-to-use prediction tools

**Enjoy classifying waste! 🌍♻️**
