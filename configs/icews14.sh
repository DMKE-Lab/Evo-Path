#!/bin/bash

data_input_dir="datasets/data_preprocessed/icews14/"
vocab_dir="datasets/data_preprocessed/icews14/vocab"
total_iterations=3000
eval_every=1000
path_length=1
hidden_size=50
embedding_size=50
batch_size=128
beta=0.05
Lambda=0.05
use_entity_embeddings=1
train_entity_embeddings=1
train_relation_embeddings=1
base_output_dir="output/icews14/"
load_model=0
model_load_dir="null"
nell_evaluation=0
