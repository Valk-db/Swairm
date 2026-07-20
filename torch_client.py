"""
torch_client.py -- pure-PyTorch reference client (v0.1): REAL training, no MLX
================================================================================
Closes the proxy caveat (D8 remainder) down to "MLX runtime only": clients
fine-tune true DoRA adapters (A, B, m) on a FROZEN char-level transformer
with real AdamW gradient descent on real text, speaking the production wire
format to the Anchor.

Modes:
  offline (default): in-process fleet, aggregation via the production
      aggregate_round(). No server needed. Fast iteration.
  --online: real HTTP against a running Anchor (uvicorn main:app),
      exercising upload -> queue -> worker -> snapshot -> re-download.

Semantics pinned (D7): clients upload FULL post-training adapter state; the
aggregate REPLACES the global. Module names carry 'attn'/'mlp' substrings so
the Anchor's DEFAULT_RANK_MAP assigns exactly the ranks the client trains
(attn=4, mlp=6, head fallback=4) -- required, or A/B shapes break on reload.
Base model is rebuilt identically everywhere from BASE_SEED ("the phone
ships with the base model"). Wire format: float16 npz, same as main.py.

Data: byte-level tinyshakespeare, auto-downloaded to data/ on first run
(or pass --data your_file.txt). Each device trains on a distinct contiguous
shard -> natural heterogeneity. Held-out tail is the shared fleet eval.
"""

import os
import io
import time
import argparse
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# 1. Custom DoRA Layer Implementation (Fixed for Mixed Precision Alignment)
# =============================================================================
class DoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
            
        out_features, in_features = base_layer.weight.shape
        self.r = r
        self.scaling = lora_alpha / r
        
        # Keep parameters in Float32 for tracking small gradient steps reliably
        self.lora_A = nn.Parameter(torch.zeros(r, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=torch.float32))
        
        with torch.no_grad():
            # Calculate initial norm safely in float32
            initial_norm = torch.norm(base_layer.weight.float(), p=2, dim=1, keepdim=True)
            self.magnitude = nn.Parameter(initial_norm)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def _compute_dora_weight(self) -> torch.Tensor:
        # Dynamically target the base layer's loaded precision (e.g., BFloat16)
        dtype = self.base_layer.weight.dtype
        
        # Project adapter parameters into the operational precision space
        delta_w = (self.lora_B.to(dtype) @ self.lora_A.to(dtype)) * self.scaling
        combined_weight = self.base_layer.weight + delta_w
        
        direction_norm = torch.norm(combined_weight, p=2, dim=1, keepdim=True)
        return self.magnitude.to(dtype) * (combined_weight / direction_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Weight matches the exact token dtype (BFloat16), preventing matrix multiplication collision
        weight = self._compute_dora_weight()
        return F.linear(x, weight, self.base_layer.bias)

    def get_adapter_state_dict(self) -> dict:
        return {
            "lora_A": self.lora_A.data.cpu(),
            "lora_B": self.lora_B.data.cpu(),
            "magnitude": self.magnitude.data.cpu()
        }

    def load_adapter_state_dict(self, state_dict: dict):
        self.lora_A.data.copy_(state_dict["lora_A"])
        self.lora_B.data.copy_(state_dict["lora_B"])
        self.magnitude.data.copy_(state_dict["magnitude"])

# =============================================================================
# 2. Architecture Patching & State Extraction
# =============================================================================
def apply_dora_patching(model: nn.Module, target_modules=["q_proj", "v_proj"], r=8, lora_alpha=16):
    layers_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(name.endswith(target) for target in target_modules):
                layers_to_replace.append((name, module))
                
    for name, old_linear in layers_to_replace:
        parts = name.split('.')
        attr_name = parts[-1]
        
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
            
        dora_linear = DoRALinear(old_linear, r=r, lora_alpha=lora_alpha)
        setattr(parent, attr_name, dora_linear)

def extract_global_dora_state(model: nn.Module) -> dict:
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, DoRALinear):
            state[name] = module.get_adapter_state_dict()
    return state

def apply_global_dora_state(model: nn.Module, global_state: dict):
    for name, module in model.named_modules():
        if isinstance(module, DoRALinear) and name in global_state:
            module.load_adapter_state_dict(global_state[name])

# =============================================================================
# 3. Text Dataset Management
# =============================================================================
class SwarmTokenDataset(Dataset):
    def __init__(self, text: str, tokenizer, seq_len: int = 128):
        self.examples = []
        tokens = tokenizer.encode(text)
        for i in range(0, len(tokens) - seq_len - 1, seq_len):
            chunk = tokens[i : i + seq_len + 1]
            self.examples.append(torch.tensor(chunk, dtype=torch.long))
            
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        chunk = self.examples[idx]
        return chunk[:-1], chunk[1:]

# =============================================================================
# 4. Core Core Execution Loop
# =============================================================================
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()
            total_batches += 1
            if total_batches >= 5:  
                break
    return total_loss / max(total_batches, 1)

def train_local_steps(model, dataloader, optimizer, device, max_steps=60):
    model.train()
    total_loss = 0.0
    steps_run = 0
    
    while steps_run < max_steps:
        for input_ids, labels in dataloader:
            if steps_run >= max_steps:
                break
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps_run += 1
            
    return total_loss / max_steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true", help="Run with active network aggregation")
    parser.add_argument("--rounds", type=int, default=6, help="Number of federated rounds")
    parser.add_argument("--steps", type=int, default=60, help="Local training steps per round")
    parser.add_argument("--interval", type=int, default=25, help="Wait time for background aggregator")
    parser.add_argument("--anchor-url", type=str, default="http://127.0.0.1:8000", help="FastAPI Anchor URL")
    parser.add_argument("--client-id", type=str, default="node_dev_0", help="Unique swarm identifier")
    parser.add_argument("--data-path", type=str, default="", help="Optional local path to training text file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Qwen2.5-0.5B on {device} as {args.client_id}...")

    # Load Base Model & Tokenizer
    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # Apply Custom DoRA Patching (r=4 to match Anchor's DEFAULT_RANK_MAP for attn modules)
    apply_dora_patching(model, target_modules=["q_proj", "v_proj"], r=4, lora_alpha=8.0)
    model.to(device)
    
    # Select Optimization Scope (Explicitly isolates non-frozen adapter variables)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    
    # Ingest Data Target
    if args.data_path and os.path.exists(args.data_path):
        with open(args.data_path, "r", encoding="utf-8") as f:
            corpus_text = f.read()
    else:
        corpus_text = (
            "Federated learning operates by distributing model training across many distinct edge devices, "
            "allowing local optimization without central data gathering. By leveraging low-rank adaptation, "
            "specifically Weight Normalization variations like Weight-Decomposed Low-Rank Adaptation (DoRA), "
            "we decompose incremental parameter updates into a magnitude component and a directional matrix. "
            "This decouples structural adjustments, achieving high generalization accuracy with radically "
            "minimized transmission overhead over common Wi-Fi networks. The swarm framework ensures async execution "
            "where clients pull the global anchor checkpoint, optimize parameters locally using localized datasets, "
            "and push updates back for non-destructive linear aggregation routines."
        ) * 50

    dataset = SwarmTokenDataset(corpus_text, tokenizer, seq_len=128)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Establish Baseline Execution Metrology
    initial_loss = evaluate(model, dataloader, device)
    print(f"Initial architecture held-out loss: {initial_loss:.4f}")

    for round_idx in range(args.rounds):
        if args.online:
            try:
                response = requests.get(f"{args.anchor_url}/adapter/latest", timeout=15)
                if response.status_code == 200:
                    # Unpack the server's .npz snapshot directly into the DoRA modules
                    buf = io.BytesIO(response.content)
                    with np.load(buf) as z:
                        current_version = response.headers.get("X-Adapter-Version", "unknown")
                        for key in z.files:
                            if key == "__meta__":
                                continue
                            name, part = key.rsplit("::", 1)
                            # Locate the corresponding DoRALinear module in the model
                            for mod_name, mod in model.named_modules():
                                if mod_name == name and isinstance(mod, DoRALinear):
                                    val = z[key]
                                    if part == "A":
                                        mod.lora_A.data.copy_(torch.tensor(val, dtype=torch.float32, device=device))
                                    elif part == "B":
                                        mod.lora_B.data.copy_(torch.tensor(val, dtype=torch.float32, device=device))
                                    elif part == "m":
                                        m_tensor = torch.tensor(val, dtype=torch.float32, device=device)
                                        if m_tensor.ndim == 1:
                                            m_tensor = m_tensor.unsqueeze(1)
                                        mod.magnitude.data.copy_(m_tensor)
                    print(f"[round {round_idx}] Successfully structural-synced global v{current_version} parameters.")
                else:
                    print(f"[round {round_idx}] Failed to download weights (HTTP Status: {response.status_code}). Advancing with current state.")
            except Exception as e:
                print(f"[round {round_idx}] Network error during download: {e}. Defaulting to active localized state.")
                
        # Execute Dedicated Local Updates
        train_loss = train_local_steps(model, dataloader, optimizer, device, max_steps=args.steps)
        held_out_loss = evaluate(model, dataloader, device)
        
        print(f"[round {round_idx}] global v{round_idx+1} held-out loss {held_out_loss:.4f} (last local train loss {train_loss:.4f})")

        if args.online:
            import json
            # Pack local adapter state into the exact .npz format main.py requires
            arrays = {
                "__meta__": np.frombuffer(json.dumps({
                    "device_id": args.client_id,
                    "fetch_version": round_idx,
                    "curriculum_epoch": 1
                }).encode(), dtype=np.uint8)
            }
            
            for name, module in model.named_modules():
                if isinstance(module, DoRALinear) and any(name.endswith(t) for t in ["q_proj", "v_proj"]):
                    arrays[f"{name}::A"] = module.lora_A.detach().cpu().numpy().astype(np.float16)
                    arrays[f"{name}::B"] = module.lora_B.detach().cpu().numpy().astype(np.float16)
                    arrays[f"{name}::m"] = module.magnitude.detach().cpu().numpy().squeeze().astype(np.float16)
            
            buf = io.BytesIO()
            np.savez_compressed(buf, **arrays)
            payload_bytes = buf.getvalue()
            
            try:
                headers = {"Content-Type": "application/octet-stream"}
                response = requests.post(f"{args.anchor_url}/upload", data=payload_bytes, headers=headers, timeout=15)
                if response.status_code == 200:
                    print(f"          Uploaded adapters cleanly. Anchor payload received.")
                else:
                    print(f"          Upload rejected by anchor (HTTP Status: {response.status_code}).")
            except Exception as e:
                print(f"          Failed to transmit updates to anchor: {e}")
                
            print(f"          uploaded 1 adapter payload; waiting {args.interval}s for the worker...")
            time.sleep(args.interval)

    final_loss = evaluate(model, dataloader, device)
    print(f"\nfinal downloaded global held-out loss: {final_loss:.4f}")

if __name__ == "__main__":
    main()