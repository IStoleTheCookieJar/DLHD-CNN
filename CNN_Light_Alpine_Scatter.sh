#!/bin/bash

#SBATCH --account=ucb450_asc2
###SBATCH --partition=aa100
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:00:00
#SBATCH --qos=normal
###SBATCH --gres=gpu
#SBATCH --job-name=CNN_Light_Scatter
#SBATCH --output=CNN_Output_Scatter_%A_%a.txt
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=kaad8904@colorado.edu
#SBATCH --array=270-271

module load anaconda
conda activate test

export myargs=$(grep ^${SLURM_ARRAY_TASK_ID} CNN_Light_Alpine_inputs.txt)

python CNN_Light_Scatter.py ${myargs} > "Trial/CNN_Output_Scatter${SLURM_ARRAY_TASK_ID}.txt"
