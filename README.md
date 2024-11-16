# Sign2German  
A Transformer-based model for translating German Sign Language (DGS) into written German text.

## Overview
This project aims to bridge communication gaps for the deaf community by creating a system that translates German Sign Language gestures into written German text. By combining MediaPipe for keypoint extraction with Transformer models for sequence recognition, this project demonstrates how natural language processing (NLP) and computer vision can enhance accessibility.

## Key Features
- **Gesture Recognition**: Uses MediaPipe to extract hand, face, and body keypoints from sign language videos.
- **Translation Model**: Implements a Transformer-based architecture to translate gesture sequences into German text.
- **End-to-End Pipeline**: Provides a complete pipeline from video input to text output, creating an efficient solution for sign language translation tasks.

## Results
The model shows promising results in translating German Sign Language into German text. Further optimization is underway to improve gesture recognition and translation accuracy.
## Repository Structure
- **`Sign2Gloss.ipynb`**: Jupyter notebook for converting sign language gestures into intermediate gloss representations.
- **`ASL2Text.ipynb`**: Jupyter notebook for converting American sign language gestures into direct to text.
- **`extract_keypoints.py`**: Python script for extracting keypoints from video frames using MediaPipe.
- **`model_code.py`**: Core model code, implementing the Transformer architecture for sequence-to-sequence translation.
- **`predictions.ipynb`**: Jupyter notebook for running predictions on new sign language inputs and generating German text outputs.

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/WMaia9/Sign2German.git
