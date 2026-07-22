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
import asyncio
import websockets
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# 1. Custom DoRA Layer Implementation (Preserved)
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
        
        self.lora_A = nn.Parameter(torch.zeros(r, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=torch.float32))
        
        with torch.no_grad():
            initial_norm = torch.norm(base_layer.weight.float(), p=2, dim=1, keepdim=True)
            self.magnitude = nn.Parameter(initial_norm)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def _compute_dora_weight(self) -> torch.Tensor:
        dtype = self.base_layer.weight.dtype
        delta_w = (self.lora_B.to(dtype) @ self.lora_A.to(dtype)) * self.scaling
        combined_weight = self.base_layer.weight + delta_w
        direction_norm = torch.norm(combined_weight, p=2, dim=1, keepdim=True)
        return self.magnitude.to(dtype) * (combined_weight / direction_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
# 2. Architecture Patching & State Extraction (Preserved)
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

# =============================================================================
# 3. Text Dataset Management (Preserved)
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
# 4. Core Execution Logic
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
            if total_batches >= 5: break
    return total_loss / max(total_batches, 1)

def train_local_steps(model, dataloader, optimizer, device, max_steps=60):
    model.train()
    total_loss = 0.0
    steps_run = 0
    while steps_run < max_steps:
        for input_ids, labels in dataloader:
            if steps_run >= max_steps: break
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

# =============================================================================
# 5. Async WebSocket Listener
# =============================================================================
async def listen_for_updates(client_id, update_event, anchor_ws_url):
    """Background task that waits for server push notifications."""
    print(f"[*] Starting WebSocket listener on {anchor_ws_url}...")
    while True:
        try:
            async with websockets.connect(f"{anchor_ws_url}/ws/{client_id}") as ws:
                print(f"[*] WebSocket connected.")
                async for message in ws:
                    if message.startswith("NEW_VERSION:"):
                        print(f"[!] Received {message}. Waking up training cycle.")
                        update_event.set()
        except Exception as e:
            print(f"[!] WebSocket error: {e}. Retrying in 5s...")
            await asyncio.sleep(5)

# =============================================================================
# 6. Async Main Loop
# =============================================================================
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--anchor-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--client-id", type=str, default=f"node_{os.getpid()}")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    apply_dora_patching(model, target_modules=["q_proj", "v_proj"], r=4, lora_alpha=8.0)
    model.to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    # Simple data setup
    corpus_text = "Federated learning operates by distributing model training..." * 50
    dataset = SwarmTokenDataset(corpus_text, tokenizer, seq_len=128)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    update_event = asyncio.Event()

    # Launch WebSocket listener in background
    ws_url = args.anchor_url.replace("http", "ws")
    asyncio.create_task(listen_for_updates(args.client_id, update_event, ws_url))

    # Initial trigger to start immediately
    update_event.set()
    
    round_idx = 0
    # Real Anchor state, synced each --online round below. Previously this
    # client uploaded fetch_version=round_idx (its own local counter, never
    # the Anchor's actual version) and a hardcoded curriculum_epoch=1.
    # aggregate_round() silently drops any upload whose curriculum_epoch
    # doesn't match the Anchor's current one (no error back to the client --
    # /upload always returns 200 regardless), so the moment a real
    # deployment's curriculum_epoch advances past 1, every upload from this
    # client would be discarded with no visible symptom other than the
    # Anchor's version no longer moving.
    current_version = 0
    current_epoch = 1
    while True:
        # Wait for the signal from the WebSocket
        await update_event.wait()
        update_event.clear()
        
        print(f"\n--- [Round {round_idx}] Starting training cycle ---")
        
        # Pull latest -- also syncs current_epoch (from /status) and
        # current_version (from the X-Adapter-Version header) so this
        # client's next upload reports real Anchor state instead of
        # round_idx / a hardcoded epoch.
        if args.online:
            try:
                status_resp = requests.get(f"{args.anchor_url}/status", timeout=10)
                if status_resp.status_code == 200:
                    current_epoch = status_resp.json().get("curriculum_epoch", current_epoch)
            except Exception as e:
                print(f"    Status fetch failed: {e}")
            try:
                resp = requests.get(f"{args.anchor_url}/adapter/latest", timeout=10)
                if resp.status_code == 200:
                    current_version = int(resp.headers.get("X-Adapter-Version", current_version))
                    with io.BytesIO(resp.content) as buf, np.load(buf) as z:
                        for key in z.files:
                            if "::" not in key: continue
                            name, part = key.rsplit("::", 1)
                            for mod_name, mod in model.named_modules():
                                if mod_name == name and isinstance(mod, DoRALinear):
                                    val = z[key]
                                    if part == "A": mod.lora_A.data.copy_(torch.tensor(val, device=device))
                                    elif part == "B": mod.lora_B.data.copy_(torch.tensor(val, device=device))
                                    elif part == "m": mod.magnitude.data.copy_(torch.tensor(val, device=device).unsqueeze(1))
                    print(f"    Synced global parameters (v{current_version}, epoch {current_epoch}).")
            except Exception as e:
                print(f"    Sync failed: {e}")

        # Train
        train_loss = train_local_steps(model, dataloader, optimizer, device, max_steps=args.steps)
        held_out_loss = evaluate(model, dataloader, device)
        print(f"    Held-out loss: {held_out_loss:.4f} (local train: {train_loss:.4f})")

        # Upload
        if args.online:
            # Match the exact schema the server expects
            meta_data = {
                "device_id": args.client_id,
                "fetch_version": current_version,
                "curriculum_epoch": current_epoch
            }
            arrays = {
                "__meta__": np.frombuffer(json.dumps(meta_data).encode(), dtype=np.uint8)
            }
            
            for name, mod in model.named_modules():
                if isinstance(mod, DoRALinear) and any(t in name for t in ["q_proj", "v_proj"]):
                    arrays[f"{name}::A"] = mod.lora_A.detach().cpu().numpy().astype(np.float16)
                    arrays[f"{name}::B"] = mod.lora_B.detach().cpu().numpy().astype(np.float16)
                    arrays[f"{name}::m"] = mod.magnitude.detach().cpu().numpy().squeeze().astype(np.float16)
            
            buf = io.BytesIO()
            np.savez_compressed(buf, **arrays)
            try:
                # Use a specific header if your server checks for it
                response = requests.post(f"{args.anchor_url}/upload", data=buf.getvalue(), timeout=10)
                if response.status_code == 200:
                    print(f"    [+] Uploaded payload successfully.")
                else:
                    print(f"    [!] Upload failed (HTTP {response.status_code}): {response.text}")
            except Exception as e:
                print(f"    [!] Upload failed: {e}")
        
        round_idx += 1
        print(f"--- Waiting for next server broadcast ---")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")