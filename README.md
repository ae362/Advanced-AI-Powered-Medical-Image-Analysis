# 🧠 Advanced AI-Powered Medical Image Analysis

### BSc Computer Science — Individual Project 2024/2025  
**Author:** Ayoub Elghayati    - Contact : elayoub407@gmail.com

**Institution:** Canterbury Christ Church University  

**Supervisor:** Ruth Thompson     - Contact : ruth.thompson1@canterbury.ac.uk

## 📝 Overview

This system is an AI-powered medical image analysis platform designed to support the diagnosis of brain tumors and cancer with high precision and full explainability. It combines advanced deep learning techniques, Grad-CAM visualizations, and GPT-4-generated medical reports to provide a seamless and intelligent diagnostic workflow.

### 🔍 Core Features

- 🧠 **AI Diagnosis** — Deep learning models (CNNs + Transformers) for classifying medical images
- 🔥 **Grad-CAM Visualizations** — Heatmaps that explain predictions
- 📝 **GPT-4 Explanation Engine** — Automatically generate human-like diagnostic reports
- 💻 **Modern Interface** — Built with Next.js, React, and Tailwind CSS
- 📤 **Image Upload** — Easy drag-and-drop for DICOM, PNG, or JPG images
- 📧 **Email Notifications** — Sends diagnostic results with visuals and explanations
- 🌍 **Multilingual Support** — Translations for patient-friendly accessibility

## ⚙️ Installation

### 1. Backend Setup (Django + DRF)
⚠️ Python 3.11.9 is recommended ⚠️

```bash
cd Backend
pip install -r requirements.txt
python manage.py runserver
```
### 2. Frontend Setup (React + Next.js)
⚠️ New terminal to avoid interference with backend server

2.
```bash
cd frontend
npm install
npm run dev
```

## 🔐 OpenAI Integration
This system uses OpenAI's GPT-4 to generate explanations for medical predictions.

⚠️ For security reasons, the actual API key is not included in the code.
1. go to backend/important.txt
2. copy the content to backend/api/views.py ( go to line 396)
3. paste the content after the dash


## 🚀 Quickstart
Once both frontend and backend are running:

Go to http://localhost:3000

Create a patient 

Upload a medical image :
you can find test samples in : https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset  For Brain Tumour

https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images For Lung Cancer

View predictions, Grad-CAM overlays, and AI-generated explanations

Optionally send results via email



## 📊 Performance
Model	Accuracy
Brain Tumor	97.5%
Cancer Detection	94.3%
Avg. Inference	10–20s

## 🛡️ Security & Compliance
Role-based access control (JWT)

GDPR-compliant patient data handling

AES-256 encryption for sensitive records

All API endpoints protected and validated

## 📁 Architecture
```graphql
├── Backend/         # Django backend with AI services and APIs
├── frontend/        # React/Next.js frontend interface
└── README.md
```

## 🧠 Future Directions
Expand to additional conditions (e.g. stroke,)

Add real-time team collaboration tools

Begin steps toward safe, validated clinical trials

## 📄 Disclaimer
This system is currently intended for academic and research purposes. However, it has been designed with real-world clinical integration in mind and may serve as a foundation for future healthcare-ready systems. Feedback from medical professionals is welcome.

## 🙏 Acknowledgements
Special thanks to:

Ruth Thompson, Supervisor

Canterbury Christ Church University

OpenAI & PyTorch communities

