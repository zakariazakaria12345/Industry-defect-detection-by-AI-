from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# ============================================================
# 1. LOAD SAVED MODEL
# ============================================================

model_path = Path("results/All_classes/All_classes_model.keras")

# For Class1 model, use this instead:
# model_path = Path("results/Class1_only/Class1_only_model.keras")

model = tf.keras.models.load_model(model_path)

print("Model loaded successfully.")
model.summary()


# ============================================================
# 2. SETTINGS MUST BE SAME AS TRAINING
# ============================================================

img_height = 256
img_width = 256

# In flow_from_directory, folders are ordered alphabetically.
# Usually:
# defective = 0
# non_defective = 1
class_names = ["defective", "non_defective"]


# ============================================================
# 3. PREDICT ONE IMAGE
# ============================================================

def predict_image(img_path):
    img = tf.keras.utils.load_img(
        img_path,
        target_size=(img_height, img_width),
        color_mode="grayscale"
    )

    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()

    img_array = tf.keras.utils.img_to_array(img)

    # same normalization used in training
    img_array = img_array / 255.0

    # add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        predicted_class = class_names[1]
        confidence = prediction * 100
    else:
        predicted_class = class_names[0]
        confidence = (1 - prediction) * 100

    print("Raw prediction:", prediction)
    print("Predicted class:", predicted_class)
    print("Confidence:", round(confidence, 2), "%")


# ============================================================
# 4. TEST IMAGE PATH
# ============================================================

# Change this path to any image you want to test
test_image = r"C:\Users\User\Desktop\Class1\Class1\Test\0010.PNG"

predict_image(test_image)