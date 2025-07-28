# MLPerf Benchmark Results

This directory contains raw MLPerf output files and logs from benchmark runs.

## 📁 Directory Structure

```
results/
├── README.md                          # This file
├── examples/                          # Sample MLPerf outputs for reference
│   ├── sample_accuracy_run/           # Example accuracy test outputs
│   │   ├── mlperf_log_accuracy.json   # Accuracy evaluation results
│   │   ├── mlperf_log_detail.txt      # Detailed MLPerf log
│   │   ├── mlperf_log_summary.txt     # Performance summary
│   │   └── mlperf_log_trace.json      # Execution trace (if enabled)
│   └── sample_performance_run/        # Example performance test outputs
│       ├── mlperf_log_detail.txt      # Detailed performance log
│       ├── mlperf_log_summary.txt     # Performance metrics
│       └── mlperf_log_trace.json      # Execution trace
└── [your_benchmark_outputs]           # Your actual MLPerf results will appear here
```

## 🎯 Expected Outputs

When you run `python3 bin/run_benchmark.py`, the following directories and files are created:

### 1. **Result Directories**
- **Format**: `local_benchmark_YYYYMMDD_HHMMSS/`
- **Created**: Automatically for each benchmark run
- **Location**: Both here and in `official_mlperf/` directory

### 2. **MLPerf Log Files**

#### **mlperf_log_summary.txt**
Performance summary with key metrics:
```
================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Server
Mode     : PerformanceOnly
Completed samples per second    : 0.34
Completed tokens per second: 44.52
Result is : INVALID/VALID
================================================
```

#### **mlperf_log_detail.txt**
Complete MLPerf logging output with:
- Configuration parameters
- Query scheduling details
- Latency measurements
- Performance statistics

#### **mlperf_log_accuracy.json** (accuracy runs only)
JSON array with tokenized outputs for each sample:
```json
[
  {
    "qsl_idx": 0,
    "data": [token_id_1, token_id_2, ...]
  },
  ...
]
```

#### **mlperf_log_trace.json**
Execution trace file (usually empty unless tracing enabled)

## 📊 Sample Metrics (NVIDIA A30, Llama-3.1-8B)

From the example runs:

### Performance Test (20 samples)
- **Throughput**: 0.34 samples/second
- **Token Rate**: 44.52 tokens/second  
- **Mean Latency**: 53,175ms (53.18 seconds)
- **First Token Latency**: 50,386ms (50.39 seconds)
- **Per Token Time**: 21.96ms

### Accuracy Test (5 samples)
- **Duration**: 54.9 seconds
- **Status**: ✅ Completed successfully
- **Tokens Generated**: ~256 tokens per sample
- **Output Quality**: Proper formatting and content

## 🔍 Understanding the Results

### **Performance Modes**
- **PerformanceOnly**: Measures throughput and latency only
- **AccuracyOnly**: Evaluates output quality without performance constraints
- **Both**: Runs both performance and accuracy evaluation

### **Result Validity**
- **VALID**: Meets all MLPerf performance requirements
- **INVALID**: May exceed latency constraints but still provides useful metrics

### **Key Latency Metrics**
- **TTFT** (Time to First Token): How long until first token appears
- **TPOT** (Time Per Output Token): Generation speed after first token
- **Total Latency**: Complete request processing time

## 🚀 Quick Analysis

To analyze your results:

1. **Check summary**: `cat results/your_run/mlperf_log_summary.txt`
2. **View detailed logs**: `less results/your_run/mlperf_log_detail.txt`
3. **Accuracy data**: `python3 -c "import json; print(len(json.load(open('results/your_run/mlperf_log_accuracy.json'))))"`

## 📝 Troubleshooting

### **Empty Results Directory**
- Results may be in `official_mlperf/` directory
- Check for error messages in benchmark output
- Verify MLPerf completed successfully

### **Invalid Results**
- Usually due to high latency (expected for development setups)
- Performance constraints can be adjusted in `user.conf`
- Focus on throughput metrics for development

### **Missing Accuracy File**
- Only created when using `--accuracy` flag
- Accuracy evaluation significantly increases runtime
- Not required for performance testing

---

**Note**: Raw MLPerf files are automatically copied to `reports/` directory with additional analysis and visualization.