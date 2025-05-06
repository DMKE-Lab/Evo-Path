# Evo-Path
The TensorFlow implementation of Evo-Path model

Paper: Evo-Path: A Two-stage Temporal Knowledge Graph Reasoning Model and Its Application in Human Behavior Prediction

# Requirements
python == 2.7.18

numpy == 1.16.6

tensorflow == 2.1.0

datetime == 4.3

networkx == 2.0

scipy == 1.2.1

tqdm == 4.19.4

# Data
ICEWS14: ./datasets/data_preprocessed/icews14

ICEWS18-7000: ./datasets/data_preprocessed/icews18-7000

ICEWS05-15-7000: ./datasets/data_preprocessed/icews05-15-7000

GDELT: Since the dataset exceeds 25MB and cannot be uploaded directly, you can use the GDELT dataset released by RE-NET. https://github.com/INK-USC/RE-Net

# Usage
For example, you can run the following command to train and test Evo-Path on the ICEWS14 dataset.

``sh run.sh configs/icews14.sh``
