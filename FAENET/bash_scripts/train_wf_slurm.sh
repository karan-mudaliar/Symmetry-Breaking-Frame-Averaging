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

# Add the repo to PYTHONPATH to ensure modules can be found
export PYTHONPATH=$PYTHONPATH:/home/mudaliar.k/github/Symmetry-Breaking-Frame-Averaging

# Debug - list the directory structure to verify paths
echo "Checking FAENET directory structure:"
ls -la FAENET/
echo "Checking faenet module directory:"
ls -la FAENET/faenet/

# Run the training script with desired parameters
echo "Running training command..."
cd FAENET  # Change to FAENET directory first

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
  --use_mlflow \
  --mlflow_experiment_name="FAENet_WF_Predictions" \
  --run_name="WF_run_1" \
  --consistency_loss \
  --consistency_weight=0.1 \
  --consistency_norm \
  --dropout=0.15

echo "Training job completed"