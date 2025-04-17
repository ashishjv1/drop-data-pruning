#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Get current timestamp for unique log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/experiment_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

# Array of final_frac values to test
final_fracs=(0.25 0.15 0.05)

# Base directory for data
DATA_PATH="/home/a.jha/drop_data_pruning_live/"

# Main log file to track overall progress
MAIN_LOG="${LOG_DIR}/main.log"
echo "Starting experiments at $(date)" > "${MAIN_LOG}"

# Run experiments sequentially
for frac in "${final_fracs[@]}"; do
    echo "Starting experiment with final_frac = $frac" | tee -a "${MAIN_LOG}"
    
    # Run the experiment and wait for it to complete
    nohup python -m drop_data_pruning_live.main \
        --auto_config \
        --use_gpu \
        --strategy 1 \
        --model_name ResNet18 \
        --scorer_name Random \
        --quoter_name DRoP \
        --num_inits 5 \
        --dataset_name TinyImageNet \
        --data_path "$DATA_PATH" \
        --final_frac "$frac" > "${LOG_DIR}/experiment_frac${frac}.log" 2>&1

    echo "Completed experiment with final_frac = $frac at $(date)" | tee -a "${MAIN_LOG}"
    echo "----------------------------------------" | tee -a "${MAIN_LOG}"
done

echo "All experiments completed at $(date)" | tee -a "${MAIN_LOG}"
echo "Logs are saved in: ${LOG_DIR}/"
echo "To check overall progress: cat ${LOG_DIR}/main.log"
echo "To check individual experiments: cat ${LOG_DIR}/experiment_frac*.log"