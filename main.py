#!/usr/bin/env python3
"""
Brain Tumor Detection - Main Entry Point

This script provides command-line interface for training and evaluating
the brain tumor detection model using MRI images.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

from predict import check_tumor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Brain Tumor Detection using CNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing MRI images (yes/no subdirectories)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="cnn-parameters-improvement-23-0.91.model",
        help="Path to trained model file"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    
    # Image arguments
    parser.add_argument(
        "--img_size",
        type=int,
        default=240,
        help="Input image size (width=height)"
    )
    
    # Inference arguments
    parser.add_argument(
        "--predict",
        type=str,
        help="Path to image for prediction"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Prediction threshold (0.1-1.0)"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "predict"],
        default="predict",
        help="Operation mode"
    )
    
    return parser.parse_args()


def train_model(args):
    """Train the brain tumor detection model."""
    print("=== Training Mode ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Image size: {args.img_size}")
    
    # TODO: Implement training pipeline
    print("Training pipeline not yet implemented.")
    print("Use the Jupyter notebook 'Brain Tumor Detection.ipynb' for training.")
    
    return True


def evaluate_model(args):
    """Evaluate the trained model."""
    print("=== Evaluation Mode ===")
    print(f"Model: {args.model}")
    print(f"Data directory: {args.data_dir}")
    print(f"Image size: {args.img_size}")
    
    # TODO: Implement evaluation pipeline
    print("Evaluation pipeline not yet implemented.")
    print("Use the Jupyter notebook 'Brain Tumor Detection.ipynb' for evaluation.")
    
    return True


def predict_image(args):
    """Predict tumor detection for a single image."""
    print("=== Prediction Mode ===")
    
    if not args.predict:
        print("Error: --predict argument required for prediction mode")
        return False
    
    if not os.path.exists(args.predict):
        print(f"Error: Image file not found: {args.predict}")
        return False
    
    print(f"Image: {args.predict}")
    print(f"Threshold: {args.threshold}")
    
    try:
        status, probability = check_tumor(args.predict, args.threshold)
        print(f"\nPrediction Result:")
        print(f"Status: {status}")
        print(f"Probability: {probability:.3f}")
        return True
    except Exception as e:
        print(f"Error during prediction: {e}")
        return False


def main():
    """Main entry point."""
    args = parse_args()
    
    print("Brain Tumor Detection System")
    print("=" * 40)
    
    if args.mode == "train":
        success = train_model(args)
    elif args.mode == "evaluate":
        success = evaluate_model(args)
    elif args.mode == "predict":
        success = predict_image(args)
    else:
        print(f"Unknown mode: {args.mode}")
        success = False
    
    if success:
        print("\nOperation completed successfully.")
    else:
        print("\nOperation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main() 