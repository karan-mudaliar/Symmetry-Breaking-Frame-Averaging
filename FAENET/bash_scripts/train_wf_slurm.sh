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
export PYTHONPATH=/home/mudaliar.k/github/Symmetry-Breaking-Frame-Averaging:$PYTHONPATH

# For debugging, let's see what's in PYTHONPATH
echo "PYTHONPATH set to: $PYTHONPATH"

# Create a patch for the frame_averaging.py file to fix device issues
echo "Patching frame_averaging.py to fix device placement issues..."
cd /home/mudaliar.k/github/Symmetry-Breaking-Frame-Averaging/FAENET
cp faenet/frame_averaging.py faenet/frame_averaging.py.backup

# Replace the tensor creation line to specify the device
sed -i 's/plus_minus_list = \[torch\.tensor(x) for x in plus_minus_list\]/plus_minus_list = \[torch\.tensor(x, device=pos.device) for x in plus_minus_list\]/g' faenet/frame_averaging.py

# Replace torch.eye(3) with device-aware version
sed -i 's/torch\.eye(3)/torch.eye(3, device=pos.device)/g' faenet/frame_averaging.py

echo "✅ Successfully patched frame_averaging.py to fix device issues"

# Fix the train.py script to properly pass consistency_loss to train_faenet
echo "Patching train.py to properly pass consistency_loss parameter..."
cp faenet/train.py faenet/train.py.backup

# Add consistency_loss parameters to the train_faenet call in main()
sed -i '/        dropout=config.dropout/ a\\        consistency_loss=config.consistency_loss,\n        consistency_weight=config.consistency_weight,\n        consistency_norm=config.consistency_norm,' faenet/train.py

echo "✅ Successfully patched train.py to pass consistency_loss parameters"

# Debug - list the directory structure to verify paths
echo "Current working directory:"
pwd
echo "FAENET directory structure:"
ls -la 
echo "faenet module directory:"
ls -la faenet/

# Run the training script with desired parameters
echo "Running training command..."

# Make sure we're in the correct directory
cd /home/mudaliar.k/github/Symmetry-Breaking-Frame-Averaging

# Show current directory for debugging
echo "Current directory before running training: $(pwd)"
echo "Directory contents:"
ls -la

# Run with full path to the script
echo "Running with direct script path..."
python -u /home/mudaliar.k/github/Symmetry-Breaking-Frame-Averaging/FAENET/faenet/train.py \
  --data_path=/home/mudaliar.k/data/DFT_data.csv \
  --structure_col=slab \
  --target_properties=[WF_top,WF_bottom] \
  --frame_averaging=2D \
  --fa_method=all \
  --batch_size=32 \
  --epochs=300 \
  --learning_rate=0.001 \
  --weight_decay=1e-5 \
  --output_dir=/home/mudaliar.k/github/Symmetry-Breaking-Frame-Averaging/outputs/WF_run_1 \
  --device=cuda \
  --consistency_loss \
  --consistency_weight=0.1 \
  --consistency_norm \
  --dropout=0.15 \
  --use_mlflow \
  --mlflow_experiment_name=FAENet_WF_Predictions \
  --run_name=WF_run_1

echo "Training job completed"