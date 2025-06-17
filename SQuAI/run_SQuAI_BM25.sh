#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=15G
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --job-name=citation-rag
#SBATCH --output=logs/slurm_%j.log

echo "Starting Citation RAG System with SLURM"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

# Set cache directories (your working setup)
export HF_DATASETS_CACHE="/data/horse/ws/jihe529c-main-rag/cache/hf_datasets"
export TRANSFORMERS_CACHE="/data/horse/ws/jihe529c-main-rag/cache/hf_models"
export HF_HOME="/data/horse/ws/jihe529c-main-rag/cache/huggingface"
export TORCH_HOME="/data/horse/ws/jihe529c-main-rag/cache/torch"
export HF_DATASETS_TRUST_REMOTE_CODE=1

echo "   Cache directories configured:"
echo "   HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "   TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "   HF_HOME: $HF_HOME"
echo "   TORCH_HOME: $TORCH_HOME"

# Create cache directories (since you cleaned them)
mkdir -p $HF_DATASETS_CACHE
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME
mkdir -p logs
mkdir -p results

echo "Directories created"

# Load modules (your working setup)
echo "Loading modules..."
module load release/24.04 GCC/12.3.0 OpenMPI/4.1.5
module load PyTorch/2.1.2

echo "Modules loaded"

# Activate your environment
echo "Activating environment..."
source bm25_env/bin/activate  # or env/bin/activate if that's your current env

echo "Environment activated"

# Install any missing dependencies (since caches are clean)
echo "Installing dependencies..."
# pip install transformers torch tqdm numpy sentence-transformers --quiet

echo "Dependencies ready"

# Run parameters for Citation RAG
MODEL="tiiuae/Falcon3-10B-Instruct"
N_VALUE=0.5
TOP_K=5
RETRIEVER_TYPE="bm25"  # or "e5" or "bm25"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# ALPHA=0.65

echo "   Run parameters:"
echo "   Model: $MODEL"
echo "   N (judge bar): $N_VALUE"
echo "   Top K: $TOP_K"
echo "   Retriever: $RETRIEVER_TYPE"

srun python run_SQuAI_BM25_backup.py \
    --model $MODEL \
    --n $N_VALUE \
    --top_k $TOP_K \
    --max_workers 6 \
    --max_context_chars 70000 \
    --data_file "dataset/deepeval_dataset.jsonl" \
    --index_dir "/data/horse/ws/inbe405h-unarxive/faiss_index" \
    --output_format jsonl \
    --output_dir "results" \
    --db_path "/data/horse/ws/jihe529c-main-rag/BM25/db2/full_text_db"

# --index_dir "/data/horse/ws/inbe405h-unarxive/test_index" \
# --data_file "dataset/sciqa_dataset.jsonl" \
# --single_question "What is machine learning and how it works?" \

echo ""
echo "Job Summary:"
echo "=============="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Output files in: results/"
echo "Logs in: logs/"

# Clean up (optional)
echo ""
echo "Cleaning up..."
# Uncomment if you want to save space by removing model cache after job
# rm -rf $TRANSFORMERS_CACHE/*  # Only if you want to clean after job

echo "Citation RAG job completed successfully!"
