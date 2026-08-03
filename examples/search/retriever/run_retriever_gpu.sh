#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2         # faiss index (fp16 ~32G) + e5 encoder both on GPU 2
source /mnt/data1/zha00175/miniconda/etc/profile.d/conda.sh
conda activate retriever
save_path=$HOME/data/searchR1
python /mnt/data1/zha00175/verl-agent/examples/search/retriever/retrieval_server.py \
  --index_path $save_path/e5_Flat.index \
  --corpus_path $save_path/wiki-18.jsonl \
  --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2 \
  --faiss_gpu --port 8010
