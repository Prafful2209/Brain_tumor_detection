import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# =====================================================
# 1. BUILD FUNCTIONAL MODEL (same as training)
# =====================================================
inputs = tf.keras.Input(shape=(128,128,3))

x = tf.keras.layers.Conv2D(32,(3,3),activation='relu',name="conv1")(inputs)
x = tf.keras.layers.MaxPooling2D(2,2,name="pool1")(x)

x = tf.keras.layers.Conv2D(64,(3,3),activation='relu',name="conv2")(x)
x = tf.keras.layers.MaxPooling2D(2,2,name="pool2")(x)

x = tf.keras.layers.Conv2D(128,(3,3),activation='relu',name="conv3")(x)
x = tf.keras.layers.MaxPooling2D(2,2,name="pool3")(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128,activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(4,activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.load_weights("best_model.h5")
model.predict(np.zeros((1,128,128,3)))

class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# =====================================================
# 2. Preprocessing
# =====================================================
def preprocess(img):
    img = cv2.resize(img, (128,128))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# =====================================================
# 3. GradCAM
# =====================================================
def gradcam(img):
    img_array = preprocess(img)
    last_conv_layer = model.get_layer("conv3")

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled, axis=-1).numpy()
    heatmap = np.maximum(heatmap,0)
    heatmap = heatmap / (heatmap.max() + 1e-8)

    heatmap = cv2.resize(heatmap, (128,128))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_resized = cv2.resize(img, (128,128))
    superimposed = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

# =====================================================
# 4. LIME
# =====================================================
def lime_explain(img):
    img_resized = cv2.resize(img, (128,128))
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) / 255.0

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        img_resized.astype("double"),
        model.predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        hide_rest=False,
        num_features=10
    )

    return mark_boundaries(temp / 255.0, mask)

# =====================================================
# 5. SHAP
# =====================================================
background = np.zeros((1,128,128,3))
explainer = shap.GradientExplainer(model, background)

def shap_explain(img):
    img_array = preprocess(img)
    shap_values = explainer.shap_values(img_array)
    shap_image = shap_values[0][0]  # first class

    # Normalize for display
    shap_image = (shap_image - shap_image.min()) / (shap_image.max() - shap_image.min() + 1e-8)
    shap_image = cv2.resize(shap_image, (128,128))
    return shap_image

# =====================================================
# 6. PREDICTION + XAI WRAPPER
# =====================================================
def analyze(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = preprocess(img_rgb)
    pred = model.predict(img_array)
    pred_class = class_names[np.argmax(pred)]

    grad = gradcam(img)
    lime_img = lime_explain(img)
    shap_img = shap_explain(img)

    return pred_class, grad, lime_img, shap_img

# =====================================================
# 7. GRADIO UI
# =====================================================

interface = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="numpy", label="Upload Brain MRI"),
    outputs=[
        gr.Textbox(label="Predicted Tumor Type"),
        gr.Image(label="GradCAM Heatmap"),
        gr.Image(label="LIME Explanation"),
        gr.Image(label="SHAP Explanation")
    ],
    title="🧠 Brain Tumor Detection + Explainable AI",
    description="Upload an MRI image to predict tumor class and visualize GradCAM, LIME, and SHAP explanations."
)

interface.launch()
