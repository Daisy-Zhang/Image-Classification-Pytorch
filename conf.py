import os

IMAGE_SIZE = 256

NUM_WORKERS = 4

TRAINING_BATCH_SIZE = 8
VAL_BATCH_SIZE = 8

EPOCH = 20
MILESTONES = [5, 10, 15]
SAVE_EPOCH = 5
WARM_EPOCH = 1

CHECKPOINTS_PATH = './checkpoints/'

LEARNING_RATE = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
GAMMA = 0.2

FRAME_SAMPLE = 10