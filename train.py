# ============================================================
# Industrial Defect Detection using CNN on DAGM Dataset
# Correct version with label matching by image number
# Student: Zakaria Zakaria
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os  # Used to search folders and files
import re  # Used to extract numbers from image names like 0837.png
import shutil  # Used to copy/delete files and folders
from pathlib import Path  # Used to write clean file paths
from collections import Counter  # Used to count images in each class

import numpy as np  # Used for arrays and numerical operations
import pandas as pd  # Used to save results in CSV tables
import matplotlib.pyplot as plt  # Used to draw graphs and confusion matrix
import tensorflow as tf  # Used to build and train the CNN model

from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Reads images from folders
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # Stops training early and saves best model
from matplotlib import image as mpimg  # Used to display sample images


# ============================================================
# 2. PROJECT SETTINGS
# ============================================================

DESKTOP = Path.home() / "Desktop"  # Automatically gets your Desktop path

CLASS1_ROOT = DESKTOP / "Class1"  # Your Class1 folder is on Desktop

ALL_CLASSES_ROOT = DESKTOP / "DAGM_All_Classes"  # Change this if your all-classes folder has another name

RUN_MODE = "class1"  # Use "class1" first, then change to "all" later

img_height = 256  # All images will be resized to height 256
img_width = 256  # All images will be resized to width 256

batch_size = 16  # The model will train on 16 images at a time
epochs = 20  # The model will study the dataset 20 times

USE_TRAIN_AUGMENTATION = False  # Keep False first to debug cleanly
USE_CLASS_WEIGHT = False  # Keep False first; turn True later if model is biased

RESULTS_ROOT = Path("results")  # Folder where results will be saved
PREPARED_ROOT = Path("prepared_datasets")  # Folder where prepared dataset will be saved


# ============================================================
# 3. FILE AND FOLDER HELPER FUNCTIONS
# ============================================================

def get_image_files(folder):  # This function returns only image files from a folder
    allowed_extensions = [".png", ".jpg", ".jpeg", ".bmp"]  # Accepted image types

    if not folder.exists():  # If the folder does not exist
        return []  # Return an empty list

    files = []  # Create empty list to store images

    for file_path in folder.iterdir():  # Loop through files inside the folder
        if file_path.is_file():  # Make sure it is a file, not a folder
            if file_path.suffix.lower() in allowed_extensions:  # Check if it is an image file
                files.append(file_path)  # Add the image file to the list

    return sorted(files)  # Return sorted image list


def has_train_test(folder):  # This checks if a folder is a DAGM class folder
    has_train = (folder / "Train").exists()  # Check if Train folder exists
    has_test = (folder / "Test").exists()  # Check if Test folder exists

    return has_train and has_test  # Return True only if both Train and Test exist


def find_dagm_class_dirs(root_folder):  # This finds all DAGM class folders automatically
    root_folder = Path(root_folder)  # Convert path to Path object

    if not root_folder.exists():  # If dataset folder does not exist
        print("ERROR: Folder does not exist:", root_folder)  # Print error
        return []  # Return empty list

    found_dirs = []  # Empty list to store class folders

    for current, dirs, files in os.walk(root_folder):  # Walk through all folders inside root
        current_path = Path(current)  # Convert current folder path to Path object

        if has_train_test(current_path):  # If this folder contains Train and Test
            found_dirs.append(current_path)  # Add it as a DAGM class folder

    def sort_key(path):  # Function to sort Class1, Class2, Class3 correctly
        text = str(path)  # Convert path to text

        match = re.search(r"Class\s*([0-9]+)", text, re.IGNORECASE)  # Search for Class number

        if match:  # If number found
            return int(match.group(1))  # Return class number for sorting

        return 999  # If no number found, put it at the end

    found_dirs = sorted(found_dirs, key=sort_key)  # Sort class folders by class number

    return found_dirs  # Return class folder list


def safe_delete_folder(folder):  # This deletes a folder if it exists
    if folder.exists():  # Check if folder exists
        shutil.rmtree(folder)  # Delete folder and everything inside it


# ============================================================
# 4. CORRECT LABEL MATCHING FUNCTIONS
# ============================================================

def extract_number(filename):  # Extract number from image or label name
    numbers = re.findall(r"\d+", filename)  # Find all numbers in filename

    if len(numbers) == 0:  # If no number found
        return None  # Return None

    return numbers[0]  # Return the first number


def get_label_ids(split_dir):  # Reads Label folder and returns IDs of defective images
    label_dir = split_dir / "Label"  # Label folder path

    if not label_dir.exists():  # If Label folder does not exist
        print("WARNING: Label folder not found:", label_dir)  # Print warning
        return set()  # Return empty set

    label_files = get_image_files(label_dir)  # Get all label mask images

    label_ids = set()  # Use set to store defective image IDs

    for label_file in label_files:  # Loop through label files
        label_id = extract_number(label_file.name)  # Extract number from label filename

        if label_id is not None:  # If number was found
            label_ids.add(label_id)  # Add number to defective IDs

    return label_ids  # Return all defective IDs


# ============================================================
# 5. PREPARE DATASET
# ============================================================

def prepare_dataset(experiment_name, dagm_class_dirs):  # Convert DAGM folders into CNN-ready folders
    prepared_dir = PREPARED_ROOT / experiment_name  # Output prepared dataset folder

    safe_delete_folder(prepared_dir)  # Delete old prepared dataset to start clean

    train_dir = prepared_dir / "train"  # New training folder
    validation_dir = prepared_dir / "validation"  # New validation folder

    folders_to_create = [  # List of folders we need to create
        train_dir / "defective",  # Training defective folder
        train_dir / "non_defective",  # Training non-defective folder
        validation_dir / "defective",  # Validation defective folder
        validation_dir / "non_defective"  # Validation non-defective folder
    ]

    for folder in folders_to_create:  # Loop through required folders
        folder.mkdir(parents=True, exist_ok=True)  # Create folder if it does not exist

    counts = Counter()  # Counter to count images

    debug_rows = []  # List to store debug information

    for class_dir in dagm_class_dirs:  # Loop through Class1, Class2, ...
        print("\nPreparing class folder:", class_dir)  # Show current class folder

        splits = [  # Original DAGM splits and new output splits
            (class_dir / "Train", train_dir, "train"),  # Train becomes train
            (class_dir / "Test", validation_dir, "validation")  # Test becomes validation
        ]

        for original_split_dir, output_split_dir, output_split_name in splits:  # Loop Train and Test
            label_ids = get_label_ids(original_split_dir)  # Get IDs of defective images from Label folder

            image_files = get_image_files(original_split_dir)  # Get original images directly inside Train/Test

            defective_count_in_split = 0  # Count defective images in this split
            non_defective_count_in_split = 0  # Count non-defective images in this split

            for img_path in image_files:  # Loop through original images
                image_id = extract_number(img_path.name)  # Extract image number from original image

                if image_id in label_ids:  # If image number exists in Label folder
                    target_class = "defective"  # Image is defective
                    defective_count_in_split += 1  # Add to defective count
                else:  # If image number is not in Label folder
                    target_class = "non_defective"  # Image is non-defective
                    non_defective_count_in_split += 1  # Add to non-defective count

                new_name = f"{class_dir.name}_{img_path.name}"  # Add class name to avoid duplicate names

                target_path = output_split_dir / target_class / new_name  # Final copied image path

                shutil.copy2(img_path, target_path)  # Copy original image to prepared dataset

                counts[(output_split_name, target_class)] += 1  # Count copied image

            debug_rows.append({  # Save debug information for this class and split
                "class_folder": class_dir.name,  # Class name
                "split": output_split_name,  # train or validation
                "original_images": len(image_files),  # Number of original images
                "label_masks": len(label_ids),  # Number of label masks
                "defective_detected": defective_count_in_split,  # Defective images detected
                "non_defective_detected": non_defective_count_in_split  # Non-defective images detected
            })

            print("Split:", output_split_name)  # Print split name
            print("Original images:", len(image_files))  # Print original image count
            print("Label masks:", len(label_ids))  # Print label mask count
            print("Defective detected:", defective_count_in_split)  # Print defective count
            print("Non-defective detected:", non_defective_count_in_split)  # Print non-defective count

    print("\nDataset prepared successfully for:", experiment_name)  # Final message

    print("\nFinal prepared image counts:")  # Print title

    for key, value in counts.items():  # Loop counts
        print(key, ":", value)  # Print count

    debug_df = pd.DataFrame(debug_rows)  # Convert debug list to DataFrame

    return prepared_dir, counts, debug_df  # Return prepared folder, counts, debug table


# ============================================================
# 6. SHOW SAMPLE IMAGES
# ============================================================

def show_sample_images(prepared_dir, experiment_name):  # Show example images from prepared data
    train_dir = prepared_dir / "train"  # Training folder

    defective_dir = train_dir / "defective"  # Defective folder
    non_defective_dir = train_dir / "non_defective"  # Non-defective folder

    defective_images = get_image_files(defective_dir)[:4]  # First 4 defective images
    non_defective_images = get_image_files(non_defective_dir)[:4]  # First 4 non-defective images

    if len(defective_images) == 0 or len(non_defective_images) == 0:  # If one class has no images
        print("Not enough images to show samples.")  # Print warning
        return  # Stop function

    fig = plt.figure(figsize=(12, 6))  # Create figure

    for i, img_path in enumerate(defective_images):  # Loop through defective images
        img = mpimg.imread(img_path)  # Read image

        ax = fig.add_subplot(2, 4, i + 1)  # Add subplot in first row
        ax.imshow(img, cmap="gray")  # Show image in grayscale
        ax.axis("off")  # Hide axis
        ax.set_title("Defective")  # Set title

    for i, img_path in enumerate(non_defective_images):  # Loop through non-defective images
        img = mpimg.imread(img_path)  # Read image

        ax = fig.add_subplot(2, 4, i + 5)  # Add subplot in second row
        ax.imshow(img, cmap="gray")  # Show image in grayscale
        ax.axis("off")  # Hide axis
        ax.set_title("Non-defective")  # Set title

    plt.suptitle(f"Sample Images - {experiment_name}")  # Main title
    plt.tight_layout()  # Fix spacing
    plt.show()  # Display figure


# ============================================================
# 7. CREATE CNN MODEL
# ============================================================

def create_cnn_model():  # Build the CNN model
    model = tf.keras.models.Sequential([  # Create Sequential model

        tf.keras.layers.Input(shape=(img_height, img_width, 1)),  # Input image: 256x256 grayscale

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),  # First convolution layer
        tf.keras.layers.MaxPooling2D(2, 2),  # Reduce image size by half

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),  # Second convolution layer
        tf.keras.layers.MaxPooling2D(2, 2),  # Reduce image size again

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),  # Third convolution layer
        tf.keras.layers.MaxPooling2D(2, 2),  # Reduce image size again

        tf.keras.layers.Flatten(),  # Convert feature maps to one long vector

        tf.keras.layers.Dense(128, activation="relu"),  # Dense layer for decision making
        tf.keras.layers.Dropout(0.3),  # Dropout to reduce overfitting

        tf.keras.layers.Dense(1, activation="sigmoid")  # Final output: 0 or 1
    ])

    model.compile(  # Prepare model for training
        loss="binary_crossentropy",  # Binary loss because we have 2 classes
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Adam optimizer with small learning rate
        metrics=["accuracy"]  # Show accuracy during training
    )

    return model  # Return the created model


# ============================================================
# 8. METRICS FUNCTIONS
# ============================================================

def calculate_metrics(conf_matrix, class_names):  # Calculate precision, recall, and F1-score
    rows = []  # Store metrics for each class

    total_correct = np.trace(conf_matrix)  # Sum of correct predictions
    total_samples = np.sum(conf_matrix)  # Total number of predictions

    accuracy = total_correct / total_samples if total_samples != 0 else 0  # Overall accuracy

    for i, class_name in enumerate(class_names):  # Loop over each class
        tp = conf_matrix[i, i]  # True positives
        fp = np.sum(conf_matrix[:, i]) - tp  # False positives
        fn = np.sum(conf_matrix[i, :]) - tp  # False negatives
        tn = np.sum(conf_matrix) - tp - fp - fn  # True negatives

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0  # Precision formula
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0  # Recall formula
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0  # F1 formula

        rows.append({  # Add class metrics to list
            "class": class_name,  # Class name
            "TP": int(tp),  # True positives
            "FP": int(fp),  # False positives
            "FN": int(fn),  # False negatives
            "TN": int(tn),  # True negatives
            "precision": precision,  # Precision
            "recall": recall,  # Recall
            "f1_score": f1_score  # F1-score
        })

    metrics_df = pd.DataFrame(rows)  # Convert metrics to table

    summary = {  # Summary metrics
        "accuracy": accuracy,  # Overall accuracy
        "macro_precision": metrics_df["precision"].mean(),  # Average precision
        "macro_recall": metrics_df["recall"].mean(),  # Average recall
        "macro_f1": metrics_df["f1_score"].mean()  # Average F1-score
    }

    return metrics_df, summary  # Return metrics table and summary


def plot_confusion_matrix(conf_matrix, class_names, results_dir, filename, title):  # Draw confusion matrix
    plt.figure(figsize=(7, 6))  # Create figure

    plt.imshow(conf_matrix)  # Display matrix as image

    plt.title(title)  # Set title
    plt.xlabel("Predicted Label")  # X-axis label
    plt.ylabel("True Label")  # Y-axis label

    plt.xticks(range(len(class_names)), class_names, rotation=30)  # X labels
    plt.yticks(range(len(class_names)), class_names)  # Y labels

    for i in range(len(class_names)):  # Loop rows
        for j in range(len(class_names)):  # Loop columns
            plt.text(j, i, str(conf_matrix[i, j]), ha="center", va="center")  # Write number in cell

    plt.tight_layout()  # Fix layout

    plt.savefig(results_dir / filename)  # Save matrix image

    plt.show()  # Show matrix


def plot_training_curves(history, results_dir):  # Plot accuracy and loss curves
    acc = history.history["accuracy"]  # Training accuracy list
    val_acc = history.history["val_accuracy"]  # Validation accuracy list
    loss = history.history["loss"]  # Training loss list
    val_loss = history.history["val_loss"]  # Validation loss list

    epochs_range = range(len(acc))  # Epoch numbers

    plt.figure()  # Create new figure
    plt.plot(epochs_range, acc, "bo", label="Training accuracy")  # Plot training accuracy
    plt.plot(epochs_range, val_acc, "b", label="Validation accuracy")  # Plot validation accuracy
    plt.title("Training and Validation Accuracy")  # Graph title
    plt.xlabel("Epoch")  # X-axis
    plt.ylabel("Accuracy")  # Y-axis
    plt.legend()  # Show legend
    plt.savefig(results_dir / "accuracy_curve.png")  # Save graph
    plt.show()  # Show graph

    plt.figure()  # Create new figure
    plt.plot(epochs_range, loss, "bo", label="Training loss")  # Plot training loss
    plt.plot(epochs_range, val_loss, "b", label="Validation loss")  # Plot validation loss
    plt.title("Training and Validation Loss")  # Graph title
    plt.xlabel("Epoch")  # X-axis
    plt.ylabel("Loss")  # Y-axis
    plt.legend()  # Show legend
    plt.savefig(results_dir / "loss_curve.png")  # Save graph
    plt.show()  # Show graph


# ============================================================
# 9. THRESHOLD EVALUATION
# ============================================================

def evaluate_thresholds(model, validation_generator, class_names, results_dir):  # Try different thresholds
    validation_generator.reset()  # Reset validation generator order

    predictions = model.predict(validation_generator).reshape(-1)  # Get prediction probabilities

    true_labels = validation_generator.classes  # Get true labels from folders

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]  # Thresholds to test

    rows = []  # Store threshold results

    best_macro_f1 = -1  # Start best macro F1 as very low
    best_threshold = 0.5  # Default best threshold
    best_conf_matrix = None  # Store best confusion matrix
    best_metrics_df = None  # Store best metrics table
    best_summary = None  # Store best summary

    for threshold in thresholds:  # Loop through thresholds
        predicted_labels = (predictions > threshold).astype(int)  # Convert probabilities to class 0/1

        conf_matrix = tf.math.confusion_matrix(  # Build confusion matrix
            true_labels,  # Real labels
            predicted_labels,  # Predicted labels
            num_classes=len(class_names)  # Number of classes
        ).numpy()  # Convert TensorFlow tensor to NumPy array

        metrics_df, summary = calculate_metrics(conf_matrix, class_names)  # Calculate metrics

        defective_row = metrics_df[metrics_df["class"] == "defective"].iloc[0]  # Get defective metrics

        rows.append({  # Store threshold result
            "threshold": threshold,  # Current threshold
            "accuracy": summary["accuracy"],  # Accuracy
            "defective_precision": defective_row["precision"],  # Defective precision
            "defective_recall": defective_row["recall"],  # Defective recall
            "defective_f1": defective_row["f1_score"],  # Defective F1
            "macro_f1": summary["macro_f1"]  # Macro F1
        })

        if summary["macro_f1"] > best_macro_f1:  # If this threshold is better
            best_macro_f1 = summary["macro_f1"]  # Update best macro F1
            best_threshold = threshold  # Update best threshold
            best_conf_matrix = conf_matrix  # Save best matrix
            best_metrics_df = metrics_df  # Save best metrics
            best_summary = summary  # Save best summary

    threshold_df = pd.DataFrame(rows)  # Convert threshold results to table

    threshold_df.to_csv(results_dir / "threshold_tuning_results.csv", index=False)  # Save threshold table

    print("\nThreshold tuning results:")  # Print title
    print(threshold_df)  # Print threshold results

    print("\nBest threshold based on macro F1:", best_threshold)  # Print best threshold

    plot_confusion_matrix(  # Plot best confusion matrix
        best_conf_matrix,  # Best matrix
        class_names,  # Class names
        results_dir,  # Results folder
        "confusion_matrix_best_threshold.png",  # File name
        f"Confusion Matrix - Best Threshold {best_threshold}"  # Title
    )

    best_metrics_df.to_csv(results_dir / "metrics_per_class_best_threshold.csv", index=False)  # Save best class metrics

    pd.DataFrame([best_summary]).to_csv(results_dir / "metrics_summary_best_threshold.csv", index=False)  # Save best summary

    return best_threshold, best_conf_matrix, best_metrics_df, best_summary  # Return best results


# ============================================================
# 10. RUN ONE EXPERIMENT
# ============================================================

def run_experiment(experiment_name, root_folder):  # Runs full training and testing process
    print("\n" + "=" * 80)  # Print separator
    print("Starting experiment:", experiment_name)  # Print experiment name
    print("=" * 80)  # Print separator

    dagm_class_dirs = find_dagm_class_dirs(root_folder)  # Find DAGM class folders

    if len(dagm_class_dirs) == 0:  # If no class folders found
        print("ERROR: No DAGM class folders found.")  # Print error
        print("Checked folder:", root_folder)  # Print checked folder
        return None  # Stop experiment

    print("\nFound DAGM class folders:")  # Print title

    for folder in dagm_class_dirs:  # Loop found folders
        print(folder)  # Print folder path

    prepared_dir, counts, debug_df = prepare_dataset(experiment_name, dagm_class_dirs)  # Prepare dataset

    results_dir = RESULTS_ROOT / experiment_name  # Results folder path

    safe_delete_folder(results_dir)  # Delete old results

    results_dir.mkdir(parents=True, exist_ok=True)  # Create results folder

    debug_df.to_csv(results_dir / "dataset_debug_label_matching.csv", index=False)  # Save debug info

    counts_rows = []  # Store count rows

    for key, value in counts.items():  # Loop counts
        counts_rows.append({  # Add count row
            "split": key[0],  # train or validation
            "class": key[1],  # defective or non_defective
            "count": value  # number of images
        })

    pd.DataFrame(counts_rows).to_csv(results_dir / "dataset_counts.csv", index=False)  # Save counts

    show_sample_images(prepared_dir, experiment_name)  # Show sample images

    train_dir = prepared_dir / "train"  # Training directory
    validation_dir = prepared_dir / "validation"  # Validation directory

    if USE_TRAIN_AUGMENTATION:  # If augmentation is enabled
        train_datagen = ImageDataGenerator(  # Create augmented training generator
            rescale=1. / 255,  # Normalize pixels to 0-1
            rotation_range=8,  # Slight rotation
            width_shift_range=0.04,  # Slight horizontal shift
            height_shift_range=0.04,  # Slight vertical shift
            zoom_range=0.08,  # Slight zoom
            horizontal_flip=True,  # Flip horizontally
            vertical_flip=True,  # Flip vertically
            fill_mode="nearest"  # Fill empty pixels
        )
    else:  # If augmentation is disabled
        train_datagen = ImageDataGenerator(rescale=1. / 255)  # Only normalize training images

    validation_datagen = ImageDataGenerator(rescale=1. / 255)  # Only normalize validation images

    train_generator = train_datagen.flow_from_directory(  # Read training images from folders
        train_dir,  # Training folder
        target_size=(img_height, img_width),  # Resize images
        batch_size=batch_size,  # Batch size
        color_mode="grayscale",  # Read images as grayscale
        class_mode="binary",  # Binary classification
        shuffle=True  # Shuffle training images
    )

    validation_generator = validation_datagen.flow_from_directory(  # Read validation images
        validation_dir,  # Validation folder
        target_size=(img_height, img_width),  # Resize images
        batch_size=batch_size,  # Batch size
        color_mode="grayscale",  # Grayscale images
        class_mode="binary",  # Binary classification
        shuffle=False  # Do not shuffle validation for correct confusion matrix
    )

    print("\nClass indices:")  # Print title
    print(train_generator.class_indices)  # Example: {'defective': 0, 'non_defective': 1}

    index_to_class = {}  # Dictionary to convert index to class name

    for class_name, index in train_generator.class_indices.items():  # Loop class indices
        index_to_class[index] = class_name  # Store reverse mapping

    class_names = [index_to_class[i] for i in range(len(index_to_class))]  # Create class names list

    print("Class names:", class_names)  # Print class names

    class_weight = None  # Default no class weights

    if USE_CLASS_WEIGHT:  # If class weight enabled
        train_classes = train_generator.classes  # Get labels of training images

        unique_classes, class_counts = np.unique(train_classes, return_counts=True)  # Count each class

        class_weight = {}  # Create class weight dictionary

        for class_index, class_count in zip(unique_classes, class_counts):  # Loop classes
            class_weight[int(class_index)] = len(train_classes) / (len(unique_classes) * class_count)  # Weight formula

        print("Class weights:", class_weight)  # Print weights

    tf.keras.backend.clear_session()  # Clear old model from memory

    model = create_cnn_model()  # Create CNN model

    model.summary()  # Print model structure

    best_model_path = results_dir / f"{experiment_name}_best_model.keras"  # Best model path

    callbacks = [  # List of training callbacks
        ModelCheckpoint(  # Save best model
            filepath=str(best_model_path),  # Save path
            monitor="val_loss",  # Watch validation loss
            save_best_only=True,  # Save only best model
            mode="min",  # Lower val_loss is better
            verbose=1  # Show messages
        ),
        EarlyStopping(  # Stop if no improvement
            monitor="val_loss",  # Watch validation loss
            patience=5,  # Wait 5 epochs before stopping
            restore_best_weights=True,  # Restore best model weights
            verbose=1  # Show messages
        )
    ]

    history = model.fit(  # Train model
        train_generator,  # Training images
        epochs=epochs,  # Number of epochs
        validation_data=validation_generator,  # Validation images
        class_weight=class_weight,  # Class weights if enabled
        callbacks=callbacks,  # Early stopping and checkpoint
        verbose=1  # Show progress
    )

    history_df = pd.DataFrame(history.history)  # Convert training history to table

    history_df.to_csv(results_dir / "training_history.csv", index=False)  # Save history

    plot_training_curves(history, results_dir)  # Save and show accuracy/loss graphs

    validation_loss, validation_accuracy = model.evaluate(validation_generator)  # Evaluate model

    print("\nValidation loss:", validation_loss)  # Print validation loss
    print("Validation accuracy:", validation_accuracy)  # Print validation accuracy

    validation_generator.reset()  # Reset validation generator

    predictions = model.predict(validation_generator).reshape(-1)  # Predict validation images

    true_labels = validation_generator.classes  # Real labels

    predicted_labels_05 = (predictions > 0.5).astype(int)  # Convert predictions using threshold 0.5

    conf_matrix_05 = tf.math.confusion_matrix(  # Build confusion matrix
        true_labels,  # Real labels
        predicted_labels_05,  # Predicted labels
        num_classes=len(class_names)  # Number of classes
    ).numpy()  # Convert to NumPy

    print("\nConfusion matrix at threshold 0.5:")  # Print title
    print(conf_matrix_05)  # Print matrix

    plot_confusion_matrix(  # Plot threshold 0.5 matrix
        conf_matrix_05,  # Matrix
        class_names,  # Class names
        results_dir,  # Results folder
        "confusion_matrix_threshold_0_5.png",  # File name
        "Confusion Matrix - Threshold 0.5"  # Title
    )

    metrics_df_05, summary_05 = calculate_metrics(conf_matrix_05, class_names)  # Metrics for threshold 0.5

    metrics_df_05.to_csv(results_dir / "metrics_per_class_threshold_0_5.csv", index=False)  # Save class metrics

    pd.DataFrame([summary_05]).to_csv(results_dir / "metrics_summary_threshold_0_5.csv", index=False)  # Save summary

    best_threshold, best_conf_matrix, best_metrics_df, best_summary = evaluate_thresholds(  # Try thresholds
        model,  # Trained model
        validation_generator,  # Validation data
        class_names,  # Class names
        results_dir  # Results folder
    )

    final_model_path = results_dir / f"{experiment_name}_final_model.keras"  # Final model path

    model.save(final_model_path)  # Save final model

    print("\nBest model saved at:", best_model_path)  # Print best model path
    print("Final model saved at:", final_model_path)  # Print final model path

    experiment_summary = {  # Save experiment summary
        "experiment": experiment_name,  # Experiment name
        "number_of_dagm_class_folders": len(dagm_class_dirs),  # Number of classes used
        "validation_loss": validation_loss,  # Validation loss
        "validation_accuracy_threshold_0_5": validation_accuracy,  # Validation accuracy
        "best_threshold": best_threshold,  # Best threshold
        "best_threshold_accuracy": best_summary["accuracy"],  # Best threshold accuracy
        "best_threshold_macro_precision": best_summary["macro_precision"],  # Best macro precision
        "best_threshold_macro_recall": best_summary["macro_recall"],  # Best macro recall
        "best_threshold_macro_f1": best_summary["macro_f1"],  # Best macro F1
        "best_model_path": str(best_model_path),  # Best model path
        "final_model_path": str(final_model_path)  # Final model path
    }

    return experiment_summary  # Return summary


# ============================================================
# 11. MAIN PROGRAM
# ============================================================

all_summaries = []  # Store all experiment summaries

if RUN_MODE == "class1":  # If user wants Class1 only
    summary = run_experiment(  # Run Class1 experiment
        experiment_name="Class1_correct",  # Experiment name
        root_folder=CLASS1_ROOT  # Class1 root folder
    )

    if summary is not None:  # If experiment completed
        all_summaries.append(summary)  # Add summary to list

elif RUN_MODE == "all":  # If user wants all classes
    summary = run_experiment(  # Run all-classes experiment
        experiment_name="All_classes_correct",  # Experiment name
        root_folder=ALL_CLASSES_ROOT  # All classes root folder
    )

    if summary is not None:  # If experiment completed
        all_summaries.append(summary)  # Add summary to list

else:  # If RUN_MODE is wrong
    print("ERROR: RUN_MODE must be either 'class1' or 'all'.")  # Print error

if len(all_summaries) > 0:  # If at least one experiment completed
    comparison_df = pd.DataFrame(all_summaries)  # Convert summaries to table

    comparison_df.to_csv("comparison_results_correct.csv", index=False)  # Save comparison table

    print("\n" + "=" * 80)  # Print separator
    print("FINAL RESULTS SUMMARY")  # Print title
    print("=" * 80)  # Print separator

    print(comparison_df)  # Print final summary

    print("\nComparison saved as: comparison_results_correct.csv")  # Print file name
else:  # If no experiment completed
    print("No experiment was completed.")  # Print message