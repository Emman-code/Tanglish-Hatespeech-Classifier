import gradio as gr
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch
from torch.nn.functional import softmax

# Load model
model = XLMRobertaForSequenceClassification.from_pretrained("EmmanuelJoshua/Tanglish-HateSpeech-Model")
tokenizer = XLMRobertaTokenizer.from_pretrained("EmmanuelJoshua/Tanglish-HateSpeech-Model")
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()
    
    label = "Offensive" if pred == 1 else "Not Offensive"
    return label, round(conf, 3)

# Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, placeholder="Enter a Tanglish sentence..."),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence")
    ],
    title="🛡️ Tanglish Hate Speech Detection API",
    description="Fine-tuned XLM-RoBERTa model to detect offensive content in Tanglish."
)

demo.launch()
