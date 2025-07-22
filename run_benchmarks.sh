#!/bin/bash
#
# Auto-generated MLPerf Benchmark Runner
# Generated from configuration: config.yaml
#

echo "🚀 Starting MLPerf Benchmarks on Configured Infrastructure"
echo "=========================================================="
echo "Model: meta-llama/Llama-3.1-8B-Instruct"
echo "Scenario: Server"
echo "Dataset: CNN DailyMail (13368 samples)"
echo "GPU Nodes: 2"
echo ""


echo "🎯 Starting benchmark on gpu-node-1 (192.168.1.100)..."
ssh username@192.168.1.100 "cd ~/official_mlperf/inference/language/llama3.1-8b && \
    nohup python3 main.py \
        --scenario Server \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --total-sample-count 13368 \
        --dataset-path cnn_eval.json \
        --vllm > gpu-node-1_benchmark.log 2>&1 &"

echo "✅ Benchmark started on gpu-node-1"

echo "🎯 Starting benchmark on gpu-node-2 (192.168.1.101)..."
ssh username@192.168.1.101 "cd ~/official_mlperf/inference/language/llama3.1-8b && \
    nohup python3 main.py \
        --scenario Server \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --total-sample-count 13368 \
        --dataset-path cnn_eval.json \
        --vllm > gpu-node-2_benchmark.log 2>&1 &"

echo "✅ Benchmark started on gpu-node-2"

echo ""
echo "🎯 All benchmarks started successfully!"
echo "📊 Monitor progress with: ./monitor_benchmarks.sh watch"
echo "📋 Check status with: ./monitor_benchmarks.sh status"
echo "📈 Collect results with: ./monitor_benchmarks.sh results"
echo ""
echo "⏰ Started at: $(date)"
