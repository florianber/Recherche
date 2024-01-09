import torch
import random
from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModel
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
import spacy
import string
import time
import os
import csv
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import numpy as np