import os # We need os for walking through directories and listing files.
import re  # for find a number inside a file name, to sort class folders in the right order.
import shutil #Used to copy, move, or delete files/folders.
from pathlib import Path #to work with file path . 
from collections import Counter # used to count how many images we have in each class and split.

import numpy as np #Used for numbers, arrays, and calculations. 
import pandas as pd # for data manipulation and saving results in CSV files. Used for tables/dataframes.
import matplotlib.pyplot as plt  # for plotting graphs and showing images. Used for visualizations.
import tensorflow as tf # for building and training the CNN model. Used for machine learning.

from tensorflow.keras.preprocessing.image import ImageDataGenerator #Used to load images and augment them.
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #EarlyStopping stops training when the model is not improving anymore. ModelCheckpoint saves the best model during training.
from matplotlib import image as mpimg #Used to read images 


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

DESKTOP = Path.home() / "Desktop" # get the path of the Desktop 

# Your Class1 path:
# Desktop/Class1/Class1/Train
# Desktop/Class1/Class1/Test
CLASS1_ROOT = DESKTOP / "Class1" #This tells the code where Class1 dataset is.


ALL_CLASSES_ROOT = DESKTOP / "allclasses"  


RUN_CLASS1_EXPERIMENT = True


RUN_ALL_CLASSES_EXPERIMENT = False

# Image size
img_height = 256
img_width = 256

batch_size = 16 # this means the model will train 16 images at a time instead of all images at once . This can help with memory and can also make training faster.
epochs = 30 # it will look at the training data 30 times . 

# This is the important new solution.
# It balances defective images in the training set using augmentation.
BALANCE_TRAINING_SET = True #if True , the code will create more defective images in the training data set using augmentation 

RESULTS_ROOT = Path("results") #This is the folder where the code saves results.
PREPARED_ROOT = Path("prepared_datasets") #this is the folder where the code saves the prepared datasets  . 


# ============================================================
# 2. BASIC HELPER FUNCTIONS
# ============================================================

def get_image_files(folder): #This function gets all image files from a folder. It checks the file extensions to make sure they are images.
    allowed_extensions = [".png", ".jpg", ".jpeg", ".bmp"]
    return sorted([
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in allowed_extensions
    ]) 


def has_train_test(folder):
    return (folder / "Train").exists() and (folder / "Test").exists()


def find_dagm_class_dirs(root_folder): 
    """
    Finds all folders that contain Train and Test.

    It works for:
    Desktop/Class1/Class1/Train

    and:
    Desktop/DAGM_All_Classes/Class1/Train
    Desktop/DAGM_All_Classes/Class2/Train
    ...
    """

    root_folder = Path(root_folder)

    if not root_folder.exists():
        print("ERROR: Folder does not exist:", root_folder)
        return []

    found_dirs = []

    for current, dirs, files in os.walk(root_folder):
        current_path = Path(current)

        if has_train_test(current_path):
            found_dirs.append(current_path)

    def sort_key(path):
        text = str(path)
        match = re.search(r"Class\s*([0-9]+)", text, re.IGNORECASE)

        if match:
            return int(match.group(1))

        return 999

    found_dirs = sorted(found_dirs, key=sort_key)

    return found_dirs


def normalize_name(name):
    """
    Makes image names and label names easier to compare.

    Example:
    0001_label -> 0001
    """

    name = name.lower()
    name = name.replace("_label", "")
    name = name.replace("label", "")
    name = name.replace("_", "")
    name = name.replace("-", "")
    name = name.replace(" ", "")

    return name


def get_label_names(split_dir):
    """
    Reads the Label folder.

    In DAGM:
    Image has matching label/mask  -> defective
    Image has no matching label    -> non_defective
    """

    label_dir = split_dir / "Label"

    if not label_dir.exists():
        print("WARNING: Label folder not found:", label_dir)
        return set()

    label_files = get_image_files(label_dir)

    label_names = set()

    for label_file in label_files:
        label_names.add(normalize_name(label_file.stem))

    return label_names

 
def safe_delete_folder(folder):
    if folder.exists():
        shutil.rmtree(folder)


# ============================================================
# 3. PREPARE DATASET
# ============================================================

def prepare_dataset(experiment_name, dagm_class_dirs):
    """
    Convert original DAGM folders into this style : 

    prepared_datasets/experiment/train/defective
    prepared_datasets/experiment/train/non_defective
    prepared_datasets/experiment/validation/defective
    prepared_datasets/experiment/validation/non_defective
    """

    prepared_dir = PREPARED_ROOT / experiment_name #This creates the output folder path.

    safe_delete_folder(prepared_dir) #This deletes the old prepared folder if it already exists.
#This creates two main folders: train and validation inside the prepared folder.
    train_dir = prepared_dir / "train"
    validation_dir = prepared_dir / "validation"
#This prepares these 4 folders:
    folders_to_create = [
        train_dir / "defective",
        train_dir / "non_defective",
        validation_dir / "defective",
        validation_dir / "non_defective"
    ]
#This actually creates the folders. If they already exist, it does nothing because of exist_ok=True.
    for folder in folders_to_create:
        folder.mkdir(parents=True, exist_ok=True)

    counts = Counter() #This is used to count how many images are copied into each folder.
#This goes through each class folder.
    for class_dir in dagm_class_dirs:
        print("\nPreparing class folder:", class_dir)
#split each folder class into train and validation based on the original Train and Test folders.
        splits = [
            (class_dir / "Train", train_dir),
            (class_dir / "Test", validation_dir)
        ]

        for original_split_dir, output_split_dir in splits:
            label_names = get_label_names(original_split_dir)#This gets the names of defective images.

            image_files = get_image_files(original_split_dir)#This gets all image files from the folder.

            for img_path in image_files:#This loops over every image.
                image_name = normalize_name(img_path.stem)#means the image name without extension.img_path = 001.PNG img_path.stem = 001
#Does this image have a label/mask? if yes it is defective, if no it is non_defective.
                if image_name in label_names:
                    target_class = "defective"
                else:
                    target_class = "non_defective"

                # This changes the image name before copying.
                new_name = f"{class_dir.name}_{img_path.name}"

                target_path = output_split_dir / target_class / new_name #This decides where the image should go.
                shutil.copy2(img_path, target_path) #This copies the image from the original dataset to the new prepared dataset

                counts[(output_split_dir.name, target_class)] += 1 #So later you know how many images were copied.

    print("\nDataset prepared for:", experiment_name) #This prints the final result.
    print("Image counts:")

    for key, value in counts.items(): #This prints the number of images in each category.
        print(key, ":", value)

    return prepared_dir, counts


# ============================================================
# 4. SHOW SAMPLE IMAGES
# ============================================================

def show_sample_images(prepared_dir, experiment_name):
    train_dir = prepared_dir / "train"

    defective_dir = train_dir / "defective"
    non_defective_dir = train_dir / "non_defective"

    defective_fnames = os.listdir(defective_dir)[:4]
    non_defective_fnames = os.listdir(non_defective_dir)[:4]

    if len(defective_fnames) == 0 or len(non_defective_fnames) == 0:
        print("Not enough images to show samples.")
        return

    fig = plt.figure(figsize=(12, 6))

    for i, fname in enumerate(defective_fnames):
        img_path = defective_dir / fname
        img = mpimg.imread(img_path)

        ax = fig.add_subplot(2, 4, i + 1)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title("Defective")

    for i, fname in enumerate(non_defective_fnames):
        img_path = non_defective_dir / fname
        img = mpimg.imread(img_path)

        ax = fig.add_subplot(2, 4, i + 5)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title("Non-defective")

    plt.suptitle(f"Sample Images - {experiment_name}")
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. BALANCE TRAINING SET WITH AUGMENTATION
# ============================================================

def balance_training_set_with_augmentation(train_dir):
    """
    This is the image version of the SMOTE idea.

    We do NOT change validation/test data.
    We only create more defective images inside the training folder.

    Before:
    defective images are few
    non_defective images are many

    After:
    defective images become close to non_defective images
    """

    defective_dir = train_dir / "defective"
    non_defective_dir = train_dir / "non_defective"

    defective_images = get_image_files(defective_dir)
    non_defective_images = get_image_files(non_defective_dir)

    defective_count = len(defective_images)
    non_defective_count = len(non_defective_images)

    print("\nBefore balancing:")
    print("Defective:", defective_count)
    print("Non-defective:", non_defective_count)

    if defective_count == 0:
        print("ERROR: No defective images found. Cannot balance.")
        return

    if defective_count >= non_defective_count:
        print("No balancing needed.")
        return

    images_to_generate = non_defective_count - defective_count

    print("Generating augmented defective images:", images_to_generate)

    augment_datagen = ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.04,
        height_shift_range=0.04,
        zoom_range=0.08,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest"
    )

    generated = 0
    image_index = 0

    while generated < images_to_generate:
        img_path = defective_images[image_index % defective_count]

        img = tf.keras.utils.load_img(
            img_path,
            color_mode="grayscale",
            target_size=(img_height, img_width)
        )

        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        aug_iter = augment_datagen.flow(
            img_array,
            batch_size=1,
            save_to_dir=defective_dir,
            save_prefix="aug_defect",
            save_format="png"
        )

        next(aug_iter)

        generated += 1
        image_index += 1

        if generated % 500 == 0:
            print("Generated:", generated, "/", images_to_generate)

    defective_count_after = len(get_image_files(defective_dir))
    non_defective_count_after = len(get_image_files(non_defective_dir))

    print("\nAfter balancing:")
    print("Defective:", defective_count_after)
    print("Non-defective:", non_defective_count_after)


# ============================================================
# 6. CREATE CNN MODEL
# ============================================================

def create_cnn_model():
    """
    Improved CNN model.

    It is still like your doctor's model:
    Conv2D -> MaxPooling -> Dense -> Output

    But we add BatchNormalization and Dropout to improve training.
    """

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(img_height, img_width, 1)),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),#This normalizes the output of the convolutional layer, which can help the model train faster and better.
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 7. METRICS
# ============================================================

def calculate_metrics(conf_matrix, class_names):
    """
    Calculate accuracy, precision, recall, and F1-score from confusion matrix.
    """

    rows = []

    total_correct = np.trace(conf_matrix)
    total_samples = np.sum(conf_matrix)

    overall_accuracy = total_correct / total_samples if total_samples != 0 else 0

    for i, class_name in enumerate(class_names):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp
        tn = np.sum(conf_matrix) - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        rows.append({
            "class": class_name,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        })

    metrics_df = pd.DataFrame(rows)

    summary = {
        "accuracy": overall_accuracy,
        "macro_precision": metrics_df["precision"].mean(),#.mean() calculates the average precision across all classes, giving us the macro precision.
        "macro_recall": metrics_df["recall"].mean(),
        "macro_f1": metrics_df["f1_score"].mean()
    }

    return metrics_df, summary


def plot_confusion_matrix(conf_matrix, class_names, results_dir, filename, title):
    plt.figure(figsize=(7, 6))
    plt.imshow(conf_matrix)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(range(len(class_names)), class_names, rotation=30)
    plt.yticks(range(len(class_names)), class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(conf_matrix[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(results_dir / filename)
    plt.show()


def plot_training_curves(history, results_dir):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure()
    plt.plot(epochs_range, acc, "bo", label="Training accuracy")
    plt.plot(epochs_range, val_acc, "b", label="Validation accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(results_dir / "accuracy_curve.png")
    plt.show()

    plt.figure()
    plt.plot(epochs_range, loss, "bo", label="Training loss")
    plt.plot(epochs_range, val_loss, "b", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(results_dir / "loss_curve.png")
    plt.show()


# ============================================================
# 8. THRESHOLD TUNING 
# ============================================================

def evaluate_thresholds(model, validation_generator, class_names, results_dir):
    """
    The model outputs probability of class 1.

    Usually:
    class 0 = defective
    class 1 = non_defective

    If prediction > threshold:
        predicted class = non_defective
    else:
        predicted class = defective

    Higher threshold means:
    the model must be very sure before saying non_defective.
    This can increase defective recall.
    """

    validation_generator.reset()

    predictions = model.predict(validation_generator).reshape(-1)
    true_labels = validation_generator.classes

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    threshold_rows = []
    matrices = {}

    for threshold in thresholds:
        predicted_labels = (predictions > threshold).astype(int)

        conf_matrix = tf.math.confusion_matrix(
            true_labels,
            predicted_labels,
            num_classes=len(class_names)
        ).numpy()

        matrices[threshold] = conf_matrix

        metrics_df, summary = calculate_metrics(conf_matrix, class_names)

        defective_row = metrics_df[metrics_df["class"] == "defective"].iloc[0]
        non_defective_row = metrics_df[metrics_df["class"] == "non_defective"].iloc[0]

        threshold_rows.append({
            "threshold": threshold,
            "accuracy": summary["accuracy"],
            "defective_precision": defective_row["precision"],
            "defective_recall": defective_row["recall"],
            "defective_f1": defective_row["f1_score"],
            "non_defective_precision": non_defective_row["precision"],
            "non_defective_recall": non_defective_row["recall"],
            "non_defective_f1": non_defective_row["f1_score"],
            "macro_f1": summary["macro_f1"]
        })

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(results_dir / "threshold_tuning_results.csv", index=False)

    print("\nThreshold tuning results:")
    print(threshold_df)

    # Best threshold based on defective F1-score
    best_index = threshold_df["defective_f1"].idxmax()
    best_threshold = float(threshold_df.loc[best_index, "threshold"])

    print("\nBest threshold based on defective F1-score:", best_threshold)

    best_conf_matrix = matrices[best_threshold]

    print("\nBest threshold confusion matrix:")
    print(best_conf_matrix)

    plot_confusion_matrix(
        best_conf_matrix,
        class_names,
        results_dir,
        "confusion_matrix_best_threshold.png",
        f"Confusion Matrix - Best Threshold {best_threshold}"
    )

    metrics_df, summary = calculate_metrics(best_conf_matrix, class_names)

    metrics_df.to_csv(results_dir / "metrics_per_class_best_threshold.csv", index=False)
    pd.DataFrame([summary]).to_csv(results_dir / "metrics_summary_best_threshold.csv", index=False)

    return best_threshold, best_conf_matrix, metrics_df, summary


# ============================================================
# 9. RUN ONE EXPERIMENT
# ============================================================

def run_experiment(experiment_name, root_folder):
    print("\n" + "=" * 80)
    print("Starting experiment:", experiment_name)
    print("=" * 80)

    dagm_class_dirs = find_dagm_class_dirs(root_folder)

    if len(dagm_class_dirs) == 0:
        print("No DAGM class folders found.")
        print("Checked root folder:", root_folder)
        return None

    print("\nFound DAGM class folders:")
    for folder in dagm_class_dirs:
        print(folder)

    prepared_dir, counts = prepare_dataset(experiment_name, dagm_class_dirs)

    results_dir = RESULTS_ROOT / experiment_name
    safe_delete_folder(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    counts_rows = []

    for key, value in counts.items():
        counts_rows.append({
            "split": key[0],
            "class": key[1],
            "count": value
        })

    counts_df = pd.DataFrame(counts_rows)
    counts_df.to_csv(results_dir / "dataset_counts_before_balancing.csv", index=False)

    show_sample_images(prepared_dir, experiment_name)

    train_dir = prepared_dir / "train"
    validation_dir = prepared_dir / "validation"

    # Balance only training data
    if BALANCE_TRAINING_SET:
        balance_training_set_with_augmentation(train_dir)

    # Save counts after balancing
    after_counts = []

    for split_name, split_dir in [("train", train_dir), ("validation", validation_dir)]:
        for class_name in ["defective", "non_defective"]:
            folder = split_dir / class_name
            after_counts.append({
                "split": split_name,
                "class": class_name,
                "count": len(get_image_files(folder))
            })

    pd.DataFrame(after_counts).to_csv(results_dir / "dataset_counts_after_balancing.csv", index=False)

    # Training augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=8,
        width_shift_range=0.04,
        height_shift_range=0.04,
        zoom_range=0.08,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest"
    )

    # Validation must remain real: no augmentation
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="binary",
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="binary",
        shuffle=False
    )

    print("\nClass indices:")
    print(train_generator.class_indices)

    index_to_class = {}

    for class_name, index in train_generator.class_indices.items():
        index_to_class[index] = class_name

    class_names = [index_to_class[i] for i in range(len(index_to_class))]

    print("Class names:", class_names)

    tf.keras.backend.clear_session()

    model = create_cnn_model()
    model.summary()

    best_model_path = results_dir / f"{experiment_name}_best_model.keras"

    callbacks = [
        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
    ]

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(results_dir / "training_history.csv", index=False)

    plot_training_curves(history, results_dir)

    validation_loss, validation_accuracy = model.evaluate(validation_generator)

    print("\nValidation loss:", validation_loss)
    print("Validation accuracy:", validation_accuracy)

    # Default threshold 0.5
    validation_generator.reset()
    predictions = model.predict(validation_generator).reshape(-1)
    true_labels = validation_generator.classes

    default_predicted_labels = (predictions > 0.5).astype(int)

    default_conf_matrix = tf.math.confusion_matrix(
        true_labels,
        default_predicted_labels,
        num_classes=len(class_names)
    ).numpy()

    print("\nDefault threshold 0.5 confusion matrix:")
    print(default_conf_matrix)

    plot_confusion_matrix(
        default_conf_matrix,
        class_names,
        results_dir,
        "confusion_matrix_threshold_0_5.png",
        "Confusion Matrix - Threshold 0.5"
    )

    default_metrics_df, default_summary = calculate_metrics(default_conf_matrix, class_names)

    default_metrics_df.to_csv(results_dir / "metrics_per_class_threshold_0_5.csv", index=False)
    pd.DataFrame([default_summary]).to_csv(results_dir / "metrics_summary_threshold_0_5.csv", index=False)

    # Threshold tuning
    best_threshold, best_conf_matrix, best_metrics_df, best_summary = evaluate_thresholds(
        model,
        validation_generator,
        class_names,
        results_dir
    )

    final_model_path = results_dir / f"{experiment_name}_final_model.keras"
    model.save(final_model_path)

    print("\nBest model saved at:", best_model_path)
    print("Final model saved at:", final_model_path)

    # Test one image with best threshold
    def predict_image(img_path, threshold):
        img = tf.keras.utils.load_img(
            img_path,
            target_size=(img_height, img_width),
            color_mode="grayscale"
        )

        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"Test Image - {experiment_name}")
        plt.show()

        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]

        if prediction > threshold:
            predicted_index = 1
        else:
            predicted_index = 0

        predicted_class = index_to_class[predicted_index]

        if predicted_index == 1:
            confidence = prediction * 100
        else:
            confidence = (1 - prediction) * 100

        print("Image:", img_path)
        print("Raw prediction:", prediction)
        print("Threshold:", threshold)
        print("Predicted class:", predicted_class)
        print("Confidence:", round(confidence, 2), "%")

    defective_test_images = get_image_files(validation_dir / "defective")
    non_defective_test_images = get_image_files(validation_dir / "non_defective")

    if len(defective_test_images) > 0:
        print("\nTesting one defective image using best threshold:")
        predict_image(defective_test_images[0], best_threshold)

    if len(non_defective_test_images) > 0:
        print("\nTesting one non-defective image using best threshold:")
        predict_image(non_defective_test_images[0], best_threshold)

    experiment_summary = {
        "experiment": experiment_name,
        "number_of_dagm_class_folders": len(dagm_class_dirs),
        "validation_loss": validation_loss,
        "validation_accuracy_threshold_0_5": validation_accuracy,
        "best_threshold": best_threshold,
        "best_threshold_accuracy": best_summary["accuracy"],
        "best_threshold_macro_precision": best_summary["macro_precision"],
        "best_threshold_macro_recall": best_summary["macro_recall"],
        "best_threshold_macro_f1": best_summary["macro_f1"],
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path)
    }

    return experiment_summary


# ============================================================
# 10. RUN EXPERIMENTS
# ============================================================

all_summaries = []

if RUN_CLASS1_EXPERIMENT:
    class1_summary = run_experiment(
        experiment_name="Class1_only_balanced",
        root_folder=CLASS1_ROOT
    )

    if class1_summary is not None:
        all_summaries.append(class1_summary)


if RUN_ALL_CLASSES_EXPERIMENT:
    all_classes_summary = run_experiment(
        experiment_name="All_classes_balanced",
        root_folder=ALL_CLASSES_ROOT
    )

    if all_classes_summary is not None:
        all_summaries.append(all_classes_summary)


# ============================================================
# 11. SAVE COMPARISON
# ============================================================

if len(all_summaries) > 0:
    comparison_df = pd.DataFrame(all_summaries)
    comparison_df.to_csv("comparison_results_balanced.csv", index=False)

    print("\n" + "=" * 80)
    print("FINAL COMPARISON RESULTS")
    print("=" * 80)
    print(comparison_df)

    print("\nComparison saved as: comparison_results_balanced.csv")

else:
    print("No experiments were completed.")