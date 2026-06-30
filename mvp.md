## 1. Project Overview

Med V-Squared QA is an interactive clinical tool designed to streamline medical image analysis. Utilizing the MUMC architecturem the platform allows users to upload medical images and ask specific diagnostic questions. The system integrates voice-to-text queries, automated batch triage, and background safety screening to accelerate clinical workflows while maintaining a secure, containerized deployment.

---

## 2. Core Features

### Feature 1: Interactive Clinical VQA Inference Engine

- **Owner:** Long
- **Description:** The primary multimodal engine. A user uploads a medical image and submits a query. The LLM processes both input data and generates a response.
- **Model Usage:** Uses the PyTorch MUMC architecture to process multimodal inputs (medical image + text query) and generate clinical answers.
- **Technical Requirements:** FastAPI endpoint integration.
- **Complexity:** Complex (1.5 - 2 days)

### Feature 2: Dual-Gate Medical Guardrail (Refusal System)

- **Owner:** Nithin
- **Description:** Ensures the model only processes valid medical queries. Implements a two-step gate:
    1. A lightweight text-intent classifier rejects non-medical questions.
    2. A Softmax confidence threshold intercepts non-medical images that produce low-confidence "guesses," throwing a safe refusal message instead.
- **Model Usage:** Small intent classification model to filter queries; uses inference probabilities from the main model to determine confidence thresholds.
- **Technical Requirements:** Small intent classification model, backend routing logic, probability threshold tuning.
- **Complexity:** Complex (1.5 days)

### Feature 3: VQA-Driven Batch Triage Sorting

- **Owner:** Sathwika
- **Description:** Allows doctors to upload a batch of images (e.g., 5-10 X-rays) at once. The system automatically runs a baseline query ("Is this image normal or abnormal?") across all images and sorts the UI queue so the doctor reviews the most critical/abnormal scans first.
- **Model Usage:** Batch inference via the VQA model to classify images as normal or abnormal based on a standard query.
- **Technical Requirements:** Batch processing logic in backend, dynamic UI sorting in React.
- **Complexity:** Medium (1 day)

---

## 3. Additional Features

### Feature 1: One-Click Clinical Report Generation & Export

- **Owner:** 
- **Enhancement Value:** After querying the VQA model, the doctor needs to document the findings. This feature allows them to click a single button to compile the medical image, their query, the model's exact response, and their own manual notes into a formatted, professional PDF report ready to be saved to an Electronic Medical Record (EMR).
- **Technical Requirements:** Pure software implementation using a PDF generation library on the backend (e.g., Python `ReportLab` or `pdfkit`) or frontend (e.g., `jspdf` in React). 
- **Complexity:** Medium (1 day)


### Feature 2: Hands-Free Voice Queries via Whisper

- **Owner:** 
- **Enhancement Value:** Eliminates typing friction for doctors, allowing them to dictate queries naturally.
- **Technical Requirements:** WebRTC microphone integration on the frontend, OpenAI Whisper API.
- **Complexity:** Medium (1 day)

### Feature 3: Audio Diagnostic Readout (Voice-Out)

- **Owner:** Sathwika
- **Enhancement Value:** Completes the hands-free loop. After the model generates a text answer, this feature reads the diagnostic result out loud so the doctor never has to look away from the patient or scan.
- **Technical Requirements:** Native Web Speech API (JavaScript) or `pyttsx3/gTTS` (Python). 100% free implementation requiring no API keys or paid subscriptions.
- **Complexity:** Simple (0.5 - 1 day)

---

## 4. Feature Backlog

1. **Medical Entity Highlighting:**
    - **Description:** Use a lightweight NER (Named Entity Recognition) package to highlight complex medical jargon in the VQA's response with hover-over definitions.
    - **Priority:** Low (UI polish).

---

## 5. Technology Stack

- **Frontend:** React or Next.js (Deployed via Vercel)
- **Backend:** Python, FastAPI (Deployed via Railway or Render)
- **ML Deployment & Optimization:** PyTorch
- **Containerization:** **Docker** & Docker Compose
- **External APIs & Models:** OpenAI Whisper API, Tesseract OCR
- **External Integrations:**
    - **Voice-In**: Open-source Whisper Python package (Audio-to-Text)
    - **Voice-Out**: Web Speech API or gTTS (Text-to-Audio)