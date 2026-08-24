#!/bin/bash

#SBATCH --account=ucb450_asc2
###SBATCH --partition=aa100
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=22:00:00
###SBATCH --gres=gpu
#SBATCH --job-name=CNN_Light
#SBATCH --output=CNN_Output_Double_%A_%a.txt
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=kaad8904@colorado.edu
#SBATCH --array=250

module load anaconda
conda activate test

export myargs=$(grep ^${SLURM_ARRAY_TASK_ID} CNN_Light_Alpine_inputs.txt)

python CNN_Light_Double.py ${myargs} > "Trial/CNN_Output_Double${SLURM_ARRAY_TASK_ID}.txt"
