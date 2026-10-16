import torch


# Device
gpu_id = 2
DEVICE = torch.device(
    f"cuda:{gpu_id}"
    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count()
    else "cpu"
)

# Data
DATA_READY_ROOT = "./data_ready"
X_NPZ_EXPLICIT = ""

# Checkpoint
CKPT_EXPLICIT = "./checkpoints_rnn/rnn_gcs_kstep_age_gamma0.50_K5_epoch0600.pt"
CKPT_ROOT = "./checkpoints_rnn"

# Test loader
BATCH_SIZE = 512
NUM_WORKERS = 0
MAX_TEST_BATCHES = 0

# Micro example
PRINT_MICRO = True
EXAMPLE_TEST_INDEX = 0
EXAMPLE_T0 = 10
MICRO_SHOW_CONTEXT = True

# Baseline
KF_R_REL = 1e-2
KF_Q_REL = 1e-4
