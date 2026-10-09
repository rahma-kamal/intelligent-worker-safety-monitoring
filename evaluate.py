import os
import time
import numpy as np
import yaml
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

from data.dataset import get_dataloaders
from models import BiLSTMHARModel, TCNHARModel, TransformerHARModel

def plot_confusion_matrix(y_true, y_pred, classes, model_name, save_dir):
    """
    Renders and saves a publication-quality confusion matrix using Seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    # Elegant dark/cool theme styling for scientific publication
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=[c.replace('_', ' ') for c in classes],
        yticklabels=[c.replace('_', ' ') for c in classes],
        cbar=True, annot_kws={"size": 10}
    )
    plt.title(f"Normalized Confusion Matrix - {model_name.upper()}", fontsize=14, pad=20, fontweight='bold')
    plt.ylabel('Ground Truth Class', fontsize=12, labelpad=10)
    plt.xlabel('Predicted Class', fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved confusion matrix figure to '{fig_path}'")

def benchmark_model_latency(model, device, num_features=92, seq_len=30, num_runs=500):
    """
    Measures exact inference latency and theoretical FPS capacity on CPU/GPU.
    """
    # Create single realistic sample tensor: shape [1, 30, 46]
    dummy_input = torch.randn(1, seq_len, num_features, device=device)
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
    end_time = time.time()
    total_elapsed = end_time - start_time
    avg_latency_ms = (total_elapsed / num_runs) * 1000
    fps = 1000.0 / avg_latency_ms
    
    return avg_latency_ms, fps

def run_evaluation():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kps_dir = config['dataset']['extracted_keypoints_dir']
    classes = config['dataset']['classes']
    checkpoint_dir = config['training']['save_dir']
    
    # 1. Load validation data
    _, val_loader = get_dataloaders(kps_dir, batch_size=64)
    
    # Pack ground truths
    all_y_true = []
    for _, batch_y in val_loader:
        all_y_true.extend(batch_y.numpy())
    all_y_true = np.array(all_y_true)
    
    # Models to test
    model_types = ["bilstm", "tcn", "transformer"]
    comparative_results = []
    
    print("\n" + "="*60)
    print(" RIGOROUS COMPARATIVE EVALUATION AND PROFILING ".center(60, "="))
    print("="*60)
    
    for m_type in model_types:
        ckpt_path = os.path.join(checkpoint_dir, f"har_{m_type}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[!] Skpping evaluation for '{m_type.upper()}' - checkpoint '{ckpt_path}' not found. Please train it first.")
            continue
            
        # Instantiate correct model class
        if m_type == "bilstm":
            opts = config['models']['bilstm']
            model = BiLSTMHARModel(input_dim=92, num_classes=len(classes),
                                  hidden_size=opts['hidden_size'], num_layers=opts['num_layers'])
        elif m_type == "tcn":
            opts = config['models']['tcn']
            model = TCNHARModel(input_dim=92, num_classes=len(classes),
                                num_channels=opts['num_channels'], kernel_size=opts['kernel_size'])
        else: # transformer
            opts = config['models']['transformer']
            model = TransformerHARModel(input_dim=92, num_classes=len(classes),
                                        d_model=opts['d_model'], nhead=opts['nhead'], num_layers=opts['num_layers'])
                                        
        model.load_checkpoint(ckpt_path, device)
        model.to(device)
        model.eval()
        
        # 2a. Run validation inference
        all_y_pred = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_y_pred.extend(preds)
                
        all_y_pred = np.array(all_y_pred)
        
        # 2b. Compute Metrics
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='weighted')
        accuracy = np.mean(all_y_true == all_y_pred) * 100
        
        # 2c. Latency and FPS benchmarks
        avg_latency, fps = benchmark_model_latency(model, device)
        
        # Get parameter counts
        total_params, _ = model.get_num_parameters()
        
        # Plot confusion matrix
        plot_confusion_matrix(all_y_true, all_y_pred, classes, m_type, checkpoint_dir)
        
        comparative_results.append({
            "Model": m_type.upper(),
            "Params": total_params,
            "Accuracy": accuracy,
            "Weighted F1": f1_w,
            "Latency (ms)": avg_latency,
            "Throughput (FPS)": fps
        })
        
        # Print standard sklearn report
        print(f"\n--- Classification Report: {m_type.upper()} ---")
        print(classification_report(all_y_true, all_y_pred, target_names=[c.replace('_', ' ') for c in classes]))
        
    # 3. Print Final Markdown Comparison Table (perfect for thesis / paper)
    print("\n" + "="*80)
    print(" COMPARATIVE SUMMARY FOR GRADUATION THESIS / RESEARCH PAPER ".center(80, "="))
    print("="*80)
    print(f"| Model | Total Parameters | Validation Accuracy | Weighted F1-Score | Latency (ms) | Inference Speed (FPS) |")
    print(f"| :--- | :---: | :---: | :---: | :---: | :---: |")
    for r in comparative_results:
        print(f"| {r['Model']:11s} | {r['Params']:16,} | {r['Accuracy']:18.2f}% | {r['Weighted F1']:17.4f} | {r['Latency (ms)']:12.2f}ms | {r['Throughput (FPS)']:20.1f} FPS |")
    print("="*80)
    print("[i] All performance figures benchmarked on device:", device.type.upper())

if __name__ == "__main__":
    run_evaluation()
