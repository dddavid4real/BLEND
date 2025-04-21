#!/usr/bin/env python3
# Author: Joel Ye
# Original file available at https://github.com/snel-repo/neural-data-transformers/blob/master/src/model_registry.py
# Adapted by Zhengrui Guo
# Added models for implementation of priviledged knowledge distillation

from src.model import (
    NeuralDataTransformer,
    NeuralDataTransformerTeacher,
)
from src.lfads import LFADS, LFADS_Teacher

from src.model_baselines import (
    RatesOracle,
    RandomModel,
)

LEARNING_MODELS = {
    "NeuralDataTransformer": NeuralDataTransformer,
    "NeuralDataTransformerTeacher": NeuralDataTransformerTeacher,
    "LFADS": LFADS,
    "LFADS_Teacher": LFADS_Teacher,
}

NONLEARNING_MODELS = {
    "Oracle": RatesOracle,
    "Random": RandomModel
}

INPUT_MASKED_MODELS = {
    "NeuralDataTransformer": NeuralDataTransformer,
    "NeuralDataTransformerTeacher": NeuralDataTransformerTeacher,
    "LFADS": LFADS,
    "LFADS_Teacher": LFADS_Teacher,
}

MODELS = {**LEARNING_MODELS, **NONLEARNING_MODELS, **INPUT_MASKED_MODELS}

def is_learning_model(model_name):
    return model_name in LEARNING_MODELS

def is_input_masked_model(model_name):
    return model_name in INPUT_MASKED_MODELS

def get_model_class(model_name):
    return MODELS[model_name]