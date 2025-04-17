#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Get current timestamp for unique log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/experiment_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

# Array of final_frac values to test
final_fracs=(0.25 0.15 0.05)

# Array of datasets to test
datasets=("CIFAR10" "CIFAR100" "TinyImageNet")

# Base directory for data
DATA_PATH="/home/a.jha/drop_data_pruning_live/"

# SLURM Configuration
MEMORY="32G"  # Adjust memory as needed
TIME="24"     # 24 hours, adjust as needed

# Main log file to track job submissions
MAIN_LOG="${LOG_DIR}/main.log"
echo "Submitting jobs at $(date)" > "${MAIN_LOG}"

# Submit jobs for each dataset and final_frac value
for dataset in "${datasets[@]}"; do
    for frac in "${final_fracs[@]}"; do
        # Set model name based on dataset
        if [ "$dataset" == "CIFAR10" ] || [ "$dataset" == "CIFAR100" ]; then
            MODEL="ResNeXt"
        else
            MODEL="ResNet18"
        fi

        echo "Submitting job for dataset = $dataset, model = $MODEL, final_frac = $frac" | tee -a "${MAIN_LOG}"
        
        # Create a temporary job script
        JOB_SCRIPT="${LOG_DIR}/job_${dataset}_${frac}.sh"
        cat << EOF > "${JOB_SCRIPT}"
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=${MEMORY}
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --time=${TIME}:00:00
#SBATCH --output=${LOG_DIR}/experiment_${dataset}_frac${frac}.log
#SBATCH --error=${LOG_DIR}/experiment_${dataset}_frac${frac}.err
#SBATCH --job-name=${dataset}_${frac}
#SBATCH --comment="DroP Data Pruning Experiment"

echo "Job started at \$(date)"
echo "Running on node: \$(hostname)"
echo "GPU information:"
nvidia-smi

# Load any necessary modules
module load cuda

echo "Starting experiment with dataset = $dataset, model = $MODEL, final_frac = $frac at \$(date)"

srun python -m drop_data_pruning_live.main \
    --auto_config \
    --use_gpu \
    --strategy 1 \
    --model_name $MODEL \
    --scorer_name Random \
    --quoter_name DRoP \
    --num_inits 5 \
    --dataset_name $dataset \
    --data_path "$DATA_PATH" \
    --final_frac "$frac"

echo "Completed experiment with dataset = $dataset, model = $MODEL, final_frac = $frac at \$(date)"
EOF

        # Make the job script executable
        chmod +x "${JOB_SCRIPT}"
        
        # Submit the job and capture the job ID
        JOB_ID=$(sbatch "${JOB_SCRIPT}" | cut -d ' ' -f 4)
        echo "Submitted job ${JOB_ID} for dataset = $dataset, model = $MODEL, final_frac = $frac" | tee -a "${MAIN_LOG}"
        echo "${JOB_ID}" > "${LOG_DIR}/job_${dataset}_${frac}.id"
    done
done

echo "All jobs submitted at $(date)" | tee -a "${MAIN_LOG}"
echo "Logs will be saved in: ${LOG_DIR}/"
echo "To check job status: squeue -u $USER"
echo "To check individual experiment logs: cat ${LOG_DIR}/experiment_*_frac*.log"
echo "To check all job IDs: cat ${LOG_DIR}/job_*.id"
echo "To monitor GPU usage: watch -n 1 nvidia-smi"