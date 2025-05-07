# 🛡️ Tanglish Hate Speech Classifier

A deep learning web app to detect offensive content in Tanglish (Tamil-English mix), built using `XLM-RoBERTa` and deployed on Hugging Face Spaces.

## 🚀 Features
- Detects hate/offensive speech in Tanglish inputs
- Real-time predictions using Gradio UI
- API endpoint for integration
- Batch CSV support (in Streamlit version)
- Custom offensive/safe word handling

## 💡 Model
Fine-tuned XLM-RoBERTa on a labeled Tanglish hate speech dataset. Hosted on [Hugging Face](https://huggingface.co/EmmanuelJoshua/Tanglish-HateSpeech-Model).

## 📦 Try it Out
👉 **Demo:** [Live App on Hugging Face Spaces](https://huggingface.co/spaces/EmmanuelJoshua/Tanglish-HateSpeech-Detector)  

## 🧠 Tech Stack
- Python
- Transformers (Hugging Face)
- PyTorch
- Gradio / Streamlit
- Hugging Face Hub

## 🖼️ Preview

![app preview](assets/model_ss.png)

## 📜 License
MIT License
