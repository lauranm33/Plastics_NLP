#!/bin/bash
#SBATCH --job-name=abstract_extraction
#SBATCH --output=/scratch/lmarrlet/Thesis/Data_Extraction/logs/extraction_%A_%a.out
#SBATCH --error=/scratch/lmarrlet/Thesis/Data_Extraction/logs/extraction_%A_%a.err
#SBATCH -p general			#Partition htc, general, private
#SBATCH -q public			#QOS
#SBATCH -t 3-00:00:00			#Time in d-hh:mm:ss
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-19			# Submits 20 jobs 
#SBATCH --mail-type=ALL
#SBATCH --mail-user="%u@asu.edu"

module load mamba/latest

source activate abstract_ase

# Define batch size
BATCH_SIZE=10000
TOTAL_PAPERS=191423

# Compute start and end indices
START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
END=$((START + BATCH_SIZE))

# Ensure last batch does not exceed total papers
if [ "$END" -gt "$TOTAL_PAPERS" ]; then
    END=$TOTAL_PAPERS
fi

echo "Processing DOIs from $START to $END"
~/.conda/envs/abstract_ase/bin/python /scratch/lmarrlet/Thesis/Data_Extraction/batched_abstracts.py --start $START --end $END
