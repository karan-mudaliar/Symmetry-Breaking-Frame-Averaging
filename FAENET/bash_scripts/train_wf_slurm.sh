#!/bin/bash

#SBATCH --partition=d2r2               
#SBATCH --nodes=1                      
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=12             
#SBATCH --gres=gpu:l40s:1              
#SBATCH --mem=16G                      
#SBATCH --time=72:00:00                
#SBATCH -o output_%j.txt                    
#SBATCH -e error_%j.txt                     
#SBATCH --mail-user=mudaliar.k@northeastern.edu  
#SBATCH --mail-type=ALL                     

# Debug information
echo "Current directory: $PWD"
echo "Python version and path:"
which python
python --version

# Load required modules
module load anaconda3/2024.06
module load cuda/12.1

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate faenet

# Go to repository directory
cd /home/mudaliar.k/github/Symmetry-Breaking-Frame-Averaging

# More debug information
echo "PYTHONPATH: $PYTHONPATH"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Starting training job..."

# Run the training script with desired parameters
python -u -m faenet.train \
  --data_path=/home/mudaliar.k/data/DFT_data.csv \
  --structure_col=slab \
  --target_properties=WF_top,WF_bottom \
  --frame_averaging=2D \
  --fa_method=all \
  --batch_size=32 \
  --epochs=300 \
  --learning_rate=0.001 \
  --weight_decay=1e-5 \
  --output_dir=/home/mudaliar.k/github/Symmetry-Breaking-Frame-Averaging/outputs/WF_run_1 \
  --device=cuda \
  --use_mlflow=True \
  --mlflow_experiment_name="FAENet_WF_Predictions" \
  --run_name="WF_run_1" \
  --consistency_loss=True \
  --consistency_weight=0.1 \
  --consistency_norm=True \
  --dropout=0.15

echo "Training job completed"