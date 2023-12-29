import gradio as gr
from PIL import Image
from fastai.vision.all import *

def predict_image(image):
    learner = load_learner('export_2.pkl')
    img = PILImage.create(image)
    pred = learner.predict(img)
    return pred[0]


def create_interface():
    
    image_input = gr.Image()

    output = gr.Text()

    iface = gr.Interface(
        fn=predict_image,
        inputs=image_input,
        outputs=output,
        title="Animal Classifier",
        description="Upload an image to identify the animal class."
    )
    return iface

if __name__ == "__main__":
    iface = create_interface()
    iface.launch(share=True)


