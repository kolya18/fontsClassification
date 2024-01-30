# Модуль распознавания

import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(200, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_font(model, image_path):
    img_array = load_and_preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    probability = predictions[0][predicted_class]
    return predicted_class, probability

def predict_class(case_number):

    class_predictions = [
        "Aguante-Regular.otf",
        "AlumniSansCollegiateOne-Italic.ttf",
        "AlumniSansCollegiateOne-Regular.ttf",
        "ambidexter_regular.otf",
        "ArefRuqaaInk-Bold.ttf",
        "ArefRuqaaInk-Regular.ttf",
        "better-vcr-5.2.ttf",
        "BrassMono-Bold.ttf",
        "BrassMono-BoldItalic.ttf",
        "BrassMono-Italic.ttf",
        "BrassMono-Regular.ttf",
        "GaneshaType-Regular.ttf",
        "GhastlyPanicCyr.otf",
        "Realest-Extended.otf",
        "TanaUncialSP.otf"
    ]

    if 0 <= case_number < len(class_predictions):
        p_class = class_predictions[case_number]
        return p_class
    else:
        return "Invalid case number"


def main():
    parser = argparse.ArgumentParser(description='Font Recognition Console App')
    parser.add_argument('checkpoint_path', type=str, help='Path to the model checkpoint file')
    parser.add_argument('image_path', type=str, help='Path to the image for font recognition')
    args = parser.parse_args()

    #загрузка обученной модели
    model = tf.keras.models.load_model(args.checkpoint_path)

    #предсказание шрифта для заданного изображения
    predicted_class, probability = predict_font(model, args.image_path)
    print(f"{predicted_class}")
    print(f"{predict_class(predicted_class)}")
    print(f"{probability}")

if __name__ == "__main__":
    main()
