import torch


# Device
gpu_id = 1
DEVICE = torch.device(
    f"cuda:{gpu_id}"
    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count()
    else "cpu"
)

# Data
DATA_READY_ROOT = "./data_ready"
X_NPZ_EXPLICIT = ""
SEED = 123

# Training
BATCH_SIZE = 256
EPOCHS = 1000
LR = 2e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# RNN
RNN_TYPE = "GRU"
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.0

# Step definition
SAMPLES_PER_STEP = 5
TAPS = SAMPLES_PER_STEP
GAIN_DIM = TAPS
PHASE_DIM = TAPS
TOK_DIM = 3 * TAPS

# Token scaling and stability
GAIN_SCALE = 20.0
EPS_NORM = 1e-6

# Training augmentation
AUGMENT_RANDOM_GLOBAL_PHASE = True

# Autoregressive feedback
DETACH_PRED_INPUT = True

# K-step prediction and loss
PRED_STEPS = 5
LOSS_GAMMA = 0.5
USE_AGE_FEATURE = True
MAX_AGE_FEATURE = 32

# Logging, evaluation, and checkpoint saving
LOG_TRAIN_EVERY_STEPS = 50
EVAL_EVERY_STEPS = 500
EVAL_BATCHES = 0
SAVE_DIR = "./checkpoints_rnn"
SAVE_EVERY_EPOCHS = 20
