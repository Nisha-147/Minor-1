# ML Model for Blind Assistive System 🎙️👁️

An end‑to‑end machine learning pipeline designed to support accessibility by enabling **voice‑to‑text transcription** and **image/audio classification** for blind and visually impaired users. This project integrates **Whisper** (for speech recognition) and **PyTorch** (for vision models), with **FFmpeg** as the audio backend.

## ✨ Features
- Voice‑to‑Text transcription using [OpenAI Whisper](https://github.com/openai/whisper)
- Image classification with a ResNet‑based PyTorch model
- Text‑to‑Speech feedback (via `pyttsx3`) for blind‑assistive interaction
- Modular training pipeline with train/val/test splits
- Easy deployment on Windows, macOS, or Linux

## 📦 Installation
### Prerequisites
- Python 3.9+
- Git
- FFmpeg (required for Whisper)

### Install FFmpeg
**Windows (Chocolatey):**
```powershell
choco install ffmpeg
