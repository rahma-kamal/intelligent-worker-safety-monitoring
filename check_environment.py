import os
import sys
import time
import platform
import psutil
import torch

def print_section(title):
    print("\n" + "="*50)
    print(f" {title} ".center(50, "="))
    print("="*50)

def profile_hardware():
    print_section("SYSTEM SPECIFICATIONS")
    
    # 1. OS Info
    print(f"[*] Operating System: {platform.system()} {platform.release()} (v{platform.version()})")
    print(f"[*] Machine Type: {platform.machine()}")
    
    # 2. CPU Info
    print(f"[*] CPU: {platform.processor()}")
    print(f"[*] Physical Cores: {psutil.cpu_count(logical=False)}")
    print(f"[*] Logical Cores: {psutil.cpu_count(logical=True)}")
    
    # 3. RAM Info
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)
    available_gb = mem.available / (1024 ** 3)
    print(f"[*] Total RAM: {total_gb:.2f} GB")
    print(f"[*] Available RAM: {available_gb:.2f} GB")
    
    if total_gb < 8.0:
        print("[!] WARNING: System RAM is under 8 GB. You may encounter memory bottlenecks when running video pipelines.")
    else:
        print("[OK] System RAM is sufficient.")

def profile_deep_learning_env():
    print_section("DEEP LEARNING ENVIRONMENT")
    
    # Python Version
    print(f"[*] Python Version: {sys.version.split()[0]}")
    
    # PyTorch Version
    print(f"[*] PyTorch Version: {torch.__version__}")
    
    # CUDA/GPU Check
    cuda_available = torch.cuda.is_available()
    print(f"[*] CUDA Available: {cuda_available}")
    
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        print(f"[OK] GPU Detected: {device_name}")
        print(f"[*] Total GPU VRAM: {vram_gb:.2f} GB")
        
        if vram_gb >= 4.0:
            print("[OK] GPU VRAM is excellent for real-time YOLOv8-pose + HAR processing.")
        else:
            print("[!] GPU VRAM is low (< 4 GB). Recommend using YOLOv8 Nano (yolov8n-pose) to avoid Out-Of-Memory errors.")
        
        device = torch.device("cuda")
    else:
        print("[!] No NVIDIA GPU/CUDA detected. Running in CPU-only mode.")
        print("[i] Note: Model training is still perfectly doable on CPU. Real-time inference will be limited in FPS.")
        device = torch.device("cpu")
        
    return device

def run_tensor_benchmark(device):
    print_section("SPEED BENCHMARK TEST")
    print(f"[*] Performing matrix multiplication benchmark on {device.type.upper()}...")
    
    # Benchmark with realistic tensor operations representing HAR sequential model forwards
    # Batch: 32, Sequence Length: 30 frames, Feature Dimensions: 128
    x = torch.randn(32, 30, 128, device=device)
    linear = torch.nn.Linear(128, 64, device=device)
    rnn = torch.nn.GRU(64, 64, batch_first=True, device=device)
    
    # Warmup
    for _ in range(50):
        out, _ = rnn(linear(x))
    
    # Measure execution latency
    start_time = time.time()
    iterations = 2000
    for _ in range(iterations):
        out, _ = rnn(linear(x))
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
    end_time = time.time()
    elapsed = end_time - start_time
    avg_latency_ms = (elapsed / iterations) * 1000
    throughput = (32 * iterations) / elapsed
    
    print(f"[OK] Benchmark Completed in {elapsed:.3f} seconds.")
    print(f"[*] Average forward batch latency: {avg_latency_ms:.3f} ms")
    print(f"[*] Estimated HAR inference throughput: {throughput:.1f} sequences/sec")
    
    print_section("FINAL RECOMMENDATION")
    if device.type == 'cuda':
        print("[RECOMMENDATION: FULL GPU ACCELERATION]")
        print("Your laptop has a dedicated NVIDIA GPU. You can run worker tracking, pose estimation, and temporal HAR model training locally at maximum performance (30+ FPS) without any issues.")
    else:
        print("[RECOMMENDATION: CPU / HYBRID INFERENCE]")
        print("Since no NVIDIA GPU is detected, you can comfortably train the temporal models (BiLSTM/TCN/Transformer) locally on CPU. For raw video processing, use YOLOv8n-pose (Nano model) to maximize CPU inference performance.")

if __name__ == "__main__":
    profile_hardware()
    device = profile_deep_learning_env()
    run_tensor_benchmark(device)
