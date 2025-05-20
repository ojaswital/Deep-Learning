import numpy as np
import argparse
import logging
from model import LogisticRegression
from data_utils import load_dataset

def preprocess(X):
    # Flatten the images and normalize pixel values
    return X.reshape(X.shape[0], -1).T / 255.

def parse_args():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Logistic Regression for Cat Classification")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--iters", type=int, default=2000, help="Number of training iterations")
    parser.add_argument("--verbose", action="store_true", help="Print cost during training")
    return parser.parse_args()

def setup_logging():
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

def main():
    args = parse_args()
    setup_logging()

    # Load dataset
    logging.info("Loading dataset...")
    X_train_orig, Y_train, X_test_orig, Y_test, classes = load_dataset()

    # Preprocess dataset
    X_train = preprocess(X_train_orig)
    X_test = preprocess(X_test_orig)

    # Initialize model
    logging.info(f"Initializing model with lr={args.lr}, iters={args.iters}")
    model = LogisticRegression(learning_rate=args.lr, num_iterations=args.iters, print_cost=args.verbose)

    # Train model
    logging.info("Training model...")
    model.fit(X_train, Y_train)

    # Evaluate model
    logging.info("Evaluating model...")
    train_accuracy = model.evaluate(X_train, Y_train)
    test_accuracy = model.evaluate(X_test, Y_test)

    logging.info(f"Train Accuracy: {train_accuracy:.2f}%")
    logging.info(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
