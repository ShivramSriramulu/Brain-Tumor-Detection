# Brain Tumor Detection

A deep learning system that detects brain tumors in MRI scans using a convolutional neural network with 91% accuracy.

## Features

- Automated brain contour extraction and cropping from MRI images
- CNN-based tumor classification with configurable confidence thresholds
- Web interface for easy image upload and prediction
- Command-line inference capabilities
- Real-time probability scoring

## Tech Stack

- Python 3.7+
- TensorFlow/Keras for deep learning
- OpenCV for image processing
- Streamlit for web interface
- Scikit-learn for data preprocessing
- PIL for image handling

## Repository Structure

```
brain_tumor_detection/
├── Brain Tumor Detection.ipynb    # Training notebook
├── cnn-parameters-improvement-23-0.91.model  # Trained model
├── predict.py                     # Inference module
├── webapp.py                      # Streamlit web app
├── requirements.txt               # Dependencies
├── uploads/                       # Temporary image storage
└── .gitignore                     # Git ignore rules
```

## Setup

### Prerequisites
- Python 3.7 or higher
- 4GB+ RAM recommended

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quickstart

### Web Interface
1. Start the web application:
```bash
streamlit run webapp.py
```

2. Open browser to http://localhost:8501
3. Upload an MRI image and adjust threshold
4. Click "Predict" for tumor detection

### Command Line Interface
```bash
# Predict tumor in a single image
python main.py --mode predict --predict path/to/image.png --threshold 0.3

# Show help
python main.py --help
```

## Data

- **Task Type**: Binary classification (tumor/no tumor)
- **Input**: MRI brain scan images (240x240 pixels)
- **Dataset**: 253 brain MRI images (155 tumor, 98 non-tumor)
- **Source**: Kaggle Brain MRI Images for Brain Tumor Detection

### Expected Data Structure
```
data/
├── yes/           # Tumor images
└── no/            # Non-tumor images
```

## Training

The model was trained using the Jupyter notebook `Brain Tumor Detection.ipynb`:

- **Architecture**: Custom CNN with Conv2D, BatchNormalization, MaxPooling2D layers
- **Input Size**: 240x240x3 RGB images
- **Data Augmentation**: Applied to increase dataset from 253 to 2065 images
- **Split**: Train/validation/test split
- **Optimizer**: Adam
- **Loss**: Binary crossentropy

### Training Commands
```bash
# Using the main script (placeholder implementation)
python main.py --mode train --data_dir data --epochs 50 --batch_size 32

# Using the shell script
./scripts/train.sh

# Using the Jupyter notebook (recommended)
jupyter notebook "Brain Tumor Detection.ipynb"
```

## Evaluation & Metrics

- **Accuracy**: 91% on test set
- **Model**: Saved as `cnn-parameters-improvement-23-0.91.model`
- **Evaluation**: Run through notebook or use `predict.py` for inference

### Evaluation Commands
```bash
# Using the main script (placeholder implementation)
python main.py --mode evaluate --data_dir data

# Using the shell script
./scripts/eval.sh

# Using the Jupyter notebook (recommended)
jupyter notebook "Brain Tumor Detection.ipynb"
```

## Results

- Achieved 91% accuracy on brain tumor detection
- Automated brain contour extraction using OpenCV
- Configurable confidence thresholds for clinical applications
- Web interface for easy deployment and use

## Serving / Inference

### Web Interface
```bash
streamlit run webapp.py
```

### Command Line
```python
from predict import check_tumor
status, probability = check_tumor("path/to/image.png", threshold=0.3)
```

### API Usage
Upload MRI image through Streamlit interface and receive:
- Tumor detection status
- Confidence probability
- Adjustable threshold control

## Model Card

### Intended Use
- Medical imaging analysis for brain tumor detection
- Educational and research purposes
- Clinical decision support (with medical professional oversight)

### Limitations
- Requires high-quality MRI scans
- Limited to binary classification (tumor/no tumor)
- Not validated for all tumor types
- Should not replace professional medical diagnosis

### Ethical Considerations
- Medical AI systems require careful validation
- Results should be reviewed by qualified medical professionals
- Patient privacy and data security must be maintained
- Model bias may exist based on training data demographics

## Roadmap

- [ ] Multi-class tumor classification
- [ ] Integration with DICOM format
- [ ] Model explainability features
- [ ] Performance optimization for real-time inference
- [ ] Docker containerization
- [ ] API endpoint for external integrations
- [ ] Comprehensive validation on larger datasets
- [ ] Clinical trial integration framework

## Contributing

### Code Style
- Follow PEP 8 guidelines
- Add docstrings for functions
- Include type hints where appropriate

### Testing
- Test image preprocessing functions
- Validate model predictions
- Ensure web interface functionality

## License

MIT License - see LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{brain_tumor_detection,
  title={Brain Tumor Detection using CNN},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/brain-tumor-detection}
}
``` 