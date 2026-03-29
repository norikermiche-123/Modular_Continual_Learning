import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# --- 1. System Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
h_dim = 128
novelty_threshold = 0.5 # Threshold in centered (but not unit-scaled) space

class TB_Router(nn.Module):
    """
    Tight-Bottleneck Router.
    Learns to reconstruct the specific coordinate space of a manifold.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16), # 16-D Bottleneck
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, h_dim)
        )
    def forward(self, h):
        return self.net(h)

def get_manifold_data(task_id, samples=100):
    """Generates data from different manifolds."""
    # Task 0 is at +2.0, Task 1 is at -2.0
    center = torch.ones(h_dim) * (2.0 if task_id == 0 else -2.0)
    # 0.1 std noise creates a very specific, tight manifold
    return torch.randn(samples, h_dim) * 0.1 + center

# --- 2. Discovery Logic ---

def test_autonomous_trigger():
    print("--- Starting Autonomous Task Discovery Test (v4.0 - Centered) ---")
    
    # 1. System starts with a trained Router for Task A
    router_a = TB_Router().to(device)
    optimizer = optim.Adam(router_a.parameters(), lr=1e-3)
    
    # Calibration Phase
    task_a_data = get_manifold_data(0, 1000).to(device)
    
    # Store the manifold center (Global Mean)
    # We do NOT use std normalization here to preserve the manifold's scale signature
    h_mean = task_a_data.mean(dim=0)
    
    print("[System] Calibrating Router A on Task A manifold...")
    router_a.train()
    for epoch in range(500): # Increased epochs for deep convergence
        optimizer.zero_grad()
        # Center the data at zero
        h_centered = task_a_data - h_mean
        recon = router_a(h_centered)
        loss = F.mse_loss(recon, h_centered)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1} - Calibration MSE: {loss.item():.6f}")
    
    router_a.eval()
    print("[System] Router A is now locked to Task A center and scale.")

    # 2. Simulate an incoming stream
    print("\n[Stream] Incoming data arriving...")
    
    # Sequence: [A, A, B, B]
    tasks_to_test = [0, 0, 1, 1] 
    for i, t_id in enumerate(tasks_to_test):
        batch = get_manifold_data(t_id, 1).to(device)
        
        with torch.no_grad():
            # Apply the Task A centering to the incoming sample
            h_centered = batch - h_mean
            recon = router_a(h_centered)
            mse = F.mse_loss(recon, h_centered).item()
        
        # Decision logic
        # In this space, Task A should have MSE ~0.01, Task B should be massive (>10.0)
        is_novel = mse > novelty_threshold
        status = "RECOGNIZED (Task A)" if not is_novel else "!!! NOVELTY DETECTED !!!"
        
        print(f"Batch {i+1} (Source Task {t_id}): MSE={mse:.4f} -> {status}")
        
        if is_novel:
            print(f"   >>> ACTION: Instantiating Simultaneous Pipeline for new Task Expert.")

if __name__ == "__main__":
    test_autonomous_trigger()