import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

# --- 1. Global Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H_DIM = 128         # Standardized latent dimension for the simulation
BN_K = 12           # The verified "Tight-Bottleneck" k
LR = 1e-3
TEMP = 2.5
NOVELTY_TAU = 0.15  # Threshold for task discovery

print(f"Unified Modular CL Framework v1.0 Initialized on {device}")

# --- 2. Modular Architecture ---

class PersistentTeacher(nn.Module):
    def __init__(self, out_dim=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(H_DIM, 256), nn.ReLU(), nn.Linear(256, out_dim))
    def forward(self, h): return self.net(h)

class StudentExpert(nn.Module):
    def __init__(self, out_dim=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(H_DIM, 128), nn.ReLU(), nn.Linear(128, out_dim))
    def forward(self, h): return self.net(h)

class TBRouter(nn.Module):
    """Tight-Bottleneck Autoencoder Router."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(H_DIM, 64), nn.ReLU(), nn.Linear(64, BN_K))
        self.decoder = nn.Sequential(nn.Linear(BN_K, 64), nn.ReLU(), nn.Linear(64, H_DIM))
    def forward(self, h):
        return self.decoder(self.encoder(h))

# --- 3. Unified Brain Manager ---

class ModularBrain:
    def __init__(self):
        self.experts = {} # {task_id: student_model}
        self.routers = {} # {task_id: (router_model, mean_anchor)}
        self.teacher = PersistentTeacher().to(device)
        self.opt_t = optim.Adam(self.teacher.parameters(), lr=LR)

    def simultaneous_train(self, task_id, loader, epochs=15):
        """Implements the Triple-Loss Simultaneous Pipeline."""
        print(f"\n[Action] Instantiating Simultaneous Pipeline for Task: {task_id}")
        student = StudentExpert().to(device)
        router = TBRouter().to(device)
        
        opt_s = optim.Adam(student.parameters(), lr=LR)
        opt_r = optim.Adam(router.parameters(), lr=LR)
        
        # Calculate manifold anchor for centering
        all_h = []
        
        for epoch in range(epochs):
            for h, lbl in loader:
                h, lbl = h.to(device), lbl.to(device)
                if epoch == 0: all_h.append(h.detach())
                
                # 1. Teacher Update
                self.opt_t.zero_grad()
                t_logits = self.teacher(h)
                loss_t = F.cross_entropy(t_logits, lbl)
                loss_t.backward(); self.opt_t.step()
                
                # 2. Student Update (Live Distillation)
                with torch.no_grad():
                    soft_targets = F.softmax(self.teacher(h) / TEMP, dim=1)
                opt_s.zero_grad()
                s_logits = student(h)
                loss_s = F.kl_div(F.log_softmax(s_logits/TEMP, 1), soft_targets, reduction='batchmean')*(TEMP**2)
                loss_s.backward(); opt_s.step()
                
                # 3. Router Update (Centered)
                h_mean = h.mean(0, keepdim=True) # Simplification for live batch
                h_centered = h - h_mean
                opt_r.zero_grad()
                recon = router(h_centered)
                loss_r = F.mse_loss(recon, h_centered)
                loss_r.backward(); opt_r.step()
        
        # Final Anchor calculation for long-term retrieval
        full_h = torch.cat(all_h)
        final_mean = full_h.mean(0)
        
        self.experts[task_id] = student.eval()
        self.routers[task_id] = (router.eval(), final_mean)
        print(f"   >>> Task {task_id} consolidated and experts frozen.")

    def probe_and_route(self, h_batch):
        """The Autonomous Decision Gate."""
        if not self.routers: return "NEW", None
        
        best_mse = float('inf')
        best_id = None
        
        for t_id, (router, mean) in self.routers.items():
            with torch.no_grad():
                h_centered = h_batch - mean
                mse = F.mse_loss(router(h_centered), h_centered).item()
                if mse < best_mse:
                    best_mse = mse
                    best_id = t_id
        
        if best_mse < NOVELTY_TAU:
            return "EXISTING", best_id
        return "NEW", None

# --- 4. Simulation Execution ---

if __name__ == "__main__":
    brain = ModularBrain()
    
    # Mock Data: Task A (Center +2) and Task B (Center -2)
    data_a = torch.randn(500, H_DIM) * 0.1 + 2.0
    lbl_a = torch.randint(0, 5, (500,))
    loader_a = [(data_a[i:i+32], lbl_a[i:i+32]) for i in range(0, 500, 32)]
    
    data_b = torch.randn(500, H_DIM) * 0.1 - 2.0
    lbl_b = torch.randint(0, 5, (500,))
    loader_b = [(data_b[i:i+32], lbl_b[i:i+32]) for i in range(0, 500, 32)]
    
    # Life Cycle
    for task_name, loader in [("Sentiment", loader_a), ("Topic", loader_b)]:
        sample_h, _ = loader[0]
        decision, match = brain.probe_and_route(sample_h.to(device))
        if decision == "NEW":
            brain.simultaneous_train(task_name, loader)
            
    # Retrieval Test
    print("\n--- Final Retrieval Stress Test ---")
    test_h = torch.randn(1, H_DIM).to(device) * 0.1 + 2.0 # Task A return
    dec, match = brain.probe_and_route(test_h)
    print(f"Input: Task A Manifold -> Decision: {dec}, Result: {match}")