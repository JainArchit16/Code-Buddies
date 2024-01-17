import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import gradio as gr
from keras.applications.resnet50 import preprocess_input

labels = ["Accident", "Normal"]

def prediction(input_img):
    try:
        loaded_model = load_model("ResNet50_no_hypertuning.h5")
        resized_image = input_img.resize((224,224))
        image_array = img_to_array(resized_image)
        image_array = tf.expand_dims(image_array, 0)
        img_array = preprocess_input(image_array)
        
        predictions = loaded_model.predict(img_array).flatten()
        return {labels[i]: float(predictions[i]) for i in range(len(labels))}
    
    except Exception as e:
        return str(e)


iface = gr.Interface(
    fn=prediction,
    inputs=gr.Image(type="pil", image_mode="RGB"),
    outputs=gr.Label(num_top_classes=1)
)

if __name__ == "__main__":
    iface.launch()
