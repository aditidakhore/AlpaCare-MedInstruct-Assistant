# AlpaCare Medical Instruction Assistant

This project is an implementation of the plan to fine-tune a 7-billion parameter language model to act as a safe, non-diagnostic medical instruction assistant. The goal is to provide helpful, educational information while strictly avoiding any form of medical diagnosis or prescription.

## Project Goal

The core goal is to fine-tune the `mistralai/Mistral-7B-v0.1` model on the `lavita/AlpaCare-MedInstruct-52k` dataset using QLoRA for parameter-efficient fine-tuning. The project emphasizes safety through a multi-layered guardrail system.

## How to Reproduce this Project

1.  **Set up the Environment:** Clone this repository and install the required dependencies.
    ```bash
    git clone [https://github.com/](https://github.com/)<your-username>/AlpaCare-MedInstruct-Assistant.git
    cd AlpaCare-MedInstruct-Assistant
    pip install -r requirements.txt
    ```

2.  **Fine-Tune the Model:** Open and run the `notebooks/colab-finetune.ipynb` notebook in a Google Colab environment with an A100 GPU. This will generate the LoRA adapter artifacts.

3.  **Run Inference Demo:** Open and run the `notebooks/inference_demo.ipynb` notebook in a Google Colab environment (a T4 GPU is sufficient). This will load the fine-tuned adapter and demonstrate the guarded inference pipeline.