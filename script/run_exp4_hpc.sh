##### instructions ####
# Usage: bsub < run_exp4_hpc.sh
#######################

#!/bin/bash
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J bert_audit_exp4
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- 
# Inference is fast, but we set 4 hours to be safe (adjust to 23:59 if queuing is fine)
#BSUB -W 04:00
# request system-memory
#BSUB -R "rusage[mem=50GB]"
### -- set the email address --
#BSUB -u leiyo@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. --
#BSUB -oo results/logs/bert_audit_exp4.out
#BSUB -eo results/logs/bert_audit_exp4.err
# -- end of LSF options --

# 1. Load System Modules (Adjust versions based on your HPC availability)
# You need CUDA for the A100 to work with PyTorch
module load python3/3.9.11
module load cuda/11.8  # Ensure this matches your PyTorch version

# 2. Activate Virtual Environment
# Assuming you use conda or venv as discussed. Uncomment the one you use:

# Option A: Conda
# source ~/miniconda3/bin/activate isqed

# Option B: Venv
# source venv/bin/activate

# 3. Create Log Directory (Just in case)
mkdir -p results/logs

# 4. Debug Info (Optional, helps verify you got the GPU)
echo "Running on host: $(hostname)"
echo "GPU info:"
nvidia-smi

# 5. Run the Experiment
# Note: We run from the root directory so python can find the 'isqed' package
python3 experiments/exp4_bert_audit_disco.py