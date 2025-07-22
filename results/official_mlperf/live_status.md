# Official MLPerf Benchmark Live Status

**Generated:** Tue Jul 22 09:39:03 AM KST 2025  
**Implementation:** Official MLCommons Reference  
**Dataset:** CNN DailyMail (13,368 samples)  
**Benchmark:** Llama-3.1-8B Server Scenario

## Current Status

### jw2 (129.254.202.252)
**Status:** ✅ RUNNING
**Progress:** 124/13,368 samples (0%)
**Performance:** INFO 07-22 09:38:56 metrics.py:345] Avg prompt throughput: 203.8 tokens/s, Avg generation throughput: 17.1 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 5.7%, CPU KV cache usage: 0.0%.

### jw3 (129.254.202.253)  
**Status:** ✅ RUNNING
**Progress:** 257/13,368 samples (1%)
**Performance:** INFO 07-22 09:39:01 metrics.py:345] Avg prompt throughput: 204.3 tokens/s, Avg generation throughput: 42.1 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 5.9%, CPU KV cache usage: 0.0%.

## Official MLPerf Features

- ✅ **Official MLCommons loadgen** - Real compliance testing
- ✅ **Full CNN DailyMail dataset** - 13,368 samples (not synthetic)
- ✅ **VLLM optimization** - Production inference engine
- ✅ **ROUGE accuracy validation** - Official scoring metrics
- ✅ **Server scenario compliance** - FirstTokenComplete callbacks
- ✅ **MLPerf-compliant reporting** - Official result format

---
*This is the genuine MLCommons implementation used in official MLPerf submissions*
