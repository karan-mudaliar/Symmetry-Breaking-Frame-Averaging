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

# Ensure we're on the dev branch
git checkout dev
echo "Current git branch: $(git branch --show-current)"

# More debug information
echo "PYTHONPATH: $PYTHONPATH"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Starting training job..."

# Go into the FAENET directory
cd /home/mudaliar.k/github/Symmetry-Breaking-Frame-Averaging/FAENET

# Show directory content
echo "Directory contents:"
ls -la

# Run the training directly with command-line arguments
echo "=========================================================="
echo "STARTING TRAINING: $(date)"
echo "=========================================================="

python -m faenet.train \
  --data_path=/home/mudaliar.k/data/DFT_data.csv \
  --structure_col=slab \
  --target_properties=WF \
  --frame_averaging=2D \
  --fa_method=all \
  --device=cuda \
  --batch_size=64 \
  --epochs=300 \
  --learning_rate=0.001 \
  --weight_decay=1e-5 \
  --dropout=0.15 \
  --no-consistency-loss \
  --use_mlflow=True \
  --mlflow_experiment_name=FAENet_WF_Predictions \
  --run_name=WF_Production_Run \
  --output_dir=/home/mudaliar.k/github/Symmetry-Breaking-Frame-Averaging/outputs/WF_Production
echo "=========================================================="
echo "PYTHON SCRIPT EXECUTION COMPLETED: $(date)"
echo "=========================================================="

echo "Training job completed"