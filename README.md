# 🧠 Advanced AI-Powered Medical Image Analysis

### BSc Computer Science — Individual Project 2024/2025  
**Author:** Ayoub Elghayati  - Contact : elayoub407@gmail.com
**Institution:** Canterbury Christ Church University  
**Supervisor:** Ruth Thompson   - Contact : ruth.thompson1@canterbury.ac.uk

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

```bash
cd Backend
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```
### 2. Frontend Setup (React + Next.js)
```bash
Copy
Edit
cd frontend
npm install
npm run dev
```
## 🚀 Quickstart
Once both frontend and backend are running:

Go to http://localhost:3000

Upload a medical image

View predictions, Grad-CAM overlays, and AI-generated explanations

Optionally send results via email

## 🔐 OpenAI Integration
This system uses OpenAI's GPT-4 to generate explanations for medical predictions.

⚠️ For security reasons, the actual API key is not included.
Each user must:
1. Create an account at https://platform.openai.com
2. Generate a key under API Keys
3. head to backend/api/views.py (go to line 396)
4. include your key in api_key (dont forget the " " )

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
Copy
Edit
.
├── Backend/         # Django backend with AI services and APIs
├── frontend/        # React/Next.js frontend interface
└── README.md
```

## 🧠 Future Directions
Expand to additional conditions (e.g. stroke, lung cancer)

Add real-time team collaboration tools

Support on-device/offline model inference

Begin steps toward safe, validated clinical trials

## 📄 Disclaimer
This system is currently intended for academic and research purposes. However, it has been designed with real-world clinical integration in mind and may serve as a foundation for future healthcare-ready systems. Feedback from medical professionals is welcome.

## 🙏 Acknowledgements
Special thanks to:

Ruth Thompson, Supervisor

Canterbury Christ Church University

OpenAI & PyTorch communities

Medical imaging dataset authors and curators
