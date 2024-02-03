import gradio as gr
import torch
from utils import *
import os

model = torch.load('AlzheimerModelCPU.pth')
model.eval()

def reset():
    return None, None, None, None, None

with gr.Blocks() as demo:
    gr.HTML("""
        <h1 style="text-align: center; font-size: 50px;">
            Alzheimer Detection
        </h1>
        <p style="text-align: center;">
            Early Detection of Alzheimer's Disease: A Deep Learning Approach for Accurate Diagnosis.
        </p>
        <h3 style="text-align: left;">
            To use the demo, please follow the steps below.
        </h3>
        <ul>
            <li style="text-align: left; padding: 0px 0px 0px 30px;">
                If you want to try examples, click one of <span style="font-weight: bold;">Examples</span> images below. Then, click 
                <span style="font-weight: bold;">Submit</span>.
            </li>
            <li style="text-align: left; padding: 0px 0px 0px 30px;">
                if you don't want to try examples, upload an image and click <span style="font-weight: bold;">Submit</span>.
            </li>
            <li style="text-align: left; padding: 0px 0px 0px 30px;">
                You can adjust the <span style="font-weight: bold;">Target</span> for your desire visualization and 
                <span style="font-weight: bold;">Plot Type</span> between <span style="font-weight: bold;">withmask</span> and <span style="font-weight: bold;">withoutmask</span>
                to plot original images with or without Grad-CAM.
            </li>
            <li style="text-align: left; padding: 0px 0px 0px 30px;">
                If you want to reset all components, click <span style="font-weight: bold;">Reset All Components</span> button.
            </li>
        </ul>
        """
    )
    with gr.Row():

        with gr.Column():
            image_area = gr.Image(sources=["upload"], type="pil", scale=2, label="Upload Image")

            with gr.Row():
                choosen_plottype =gr.Radio(choices=["withmask", "withoutmask"], value="withmask", label="Plot Type", scale=1, interactive=True)
                choosen_target = gr.Slider(minimum=0, maximum=200, step=1, value=100, label="Classifier OutputTarget", scale=1, interactive=True)
                submit_btn = gr.Button("Submit", variant='primary', scale=1)
                
        with gr.Column():
            gr.HTML("""
            <h2 style="text-align: center;">
                
            </h2>
            <h2 style="text-align: center;">
                Prediction and Grad-CAM
            </h2>
            <p style="text-align: center;">
                Alzheimer detection result with Grad-CAM for 3 images. Original, White Matter and Gray Matter image.
            </p>
            """
            )
            text_area = gr.Textbox(label="Prediction Result")
            plotarea_original = gr.Plot(label="Original image")
            plotarea_white = gr.Plot(label="White Matter image")
            plotarea_gray = gr.Plot(label="Gray Matter image")
        
            reset_btn = gr.Button("Reset All Components", variant='stop', scale=1) 

    gr.HTML("""
            <h2 style="text-align: left;">
                Examples
            </h2>
            <p style="text-align: left;">
                You can select 1 image from the examples and click "Submit".
            </p>
            <p style="text-align: left;">
                These images represent 4 different stages of Alzheimer's disease in the following order:<span style="font-weight: bold;">
                Non Demented, Very Mild Demented, Mild Demented and Moderate Demented.</span>
            </p>
            """
            )
    examples = gr.Examples(examples=[   os.path.join(os.getcwd(), "non.jpg"), 
                                        os.path.join(os.getcwd(),"verymild.jpg"), 
                                        os.path.join(os.getcwd(), "mild.jpg"), 
                                        os.path.join(os.getcwd(), "moderate.jpg")], 
                                        inputs=image_area,
                                        outputs=image_area, 
                                        label="Examples",
                                        )

    submit_btn.click(lambda x, target, plot_type: predict_and_gradcam(x, model=model, target=target, plot_type=plot_type), inputs=[image_area, choosen_target, choosen_plottype], outputs=[text_area, plotarea_original, plotarea_white, plotarea_gray])
    reset_btn.click(reset, outputs=[image_area, text_area, plotarea_original, plotarea_white, plotarea_gray])

demo.launch()