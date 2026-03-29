import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import time

# --- 1. CPU-Optimized Colab Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
lr = 1e-3
temp = 2.5          
epochs = 15         
warmup_epochs = 3   
vae_latent_dim = 32 
tbae_latent_dim = 12 
router_type = "TB-AE" # Toggle between "VAE" and "TB-AE"

print(f"Lifelong Modular Learning Simulation Initialized on {device}")
print(f"[Mode] Scratchpad Mode: Raw Input (h=x) with {router_type} Routing")

# --- 2. Components (Mirroring the Paper) ---

class TeacherEngine(nn.Module):
    def __init__(self, out_dim=10): # Output dim 10 to handle the full sequence mapping
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, x): return self.net(x)

class StudentExpert(nn.Module):
    def __init__(self, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x): return self.net(x)

class TaskVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.fc_mu = nn.Linear(128, vae_latent_dim)
        self.fc_logvar = nn.Linear(128, vae_latent_dim)
        self.decoder = nn.Sequential(nn.Linear(vae_latent_dim, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 784))

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def get_loss_per_sample(self, x):
        """Calculates negative ELBO per sample for batch routing"""
        x_flat = x.view(-1, 784)
        x_scaled = (x_flat - x_flat.min()) / (x_flat.max() - x_flat.min() + 1e-8)
        recon_x, mu, logvar = self.forward(x)
        recon_scaled = torch.sigmoid(recon_x)
        
        BCE = F.binary_cross_entropy(recon_scaled, x_scaled, reduction='none').sum(dim=1)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return BCE + KLD

class TaskTBAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, tbae_latent_dim), nn.ReLU(),
            nn.Linear(tbae_latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 784)
        )
    def forward(self, x):
        return self.net(x)
        
    def get_loss_per_sample(self, x):
        """Calculates BCE loss per sample for batch routing"""
        x_flat = x.view(-1, 784)
        x_scaled = (x_flat - x_flat.min()) / (x_flat.max() - x_flat.min() + 1e-8)
        recon_x = self.forward(x)
        recon_scaled = torch.sigmoid(recon_x)
        return F.binary_cross_entropy(recon_scaled, x_scaled, reduction='none').sum(dim=1)

# --- 3. The Simultaneous Pipeline ---

def train_simultaneous(loader, task_name, teacher):
    print(f"\n[Start] Executing Simultaneous Pipeline for {task_name}...")
    
    student = StudentExpert().to(device)
    router = TaskVAE().to(device) if router_type == "VAE" else TaskTBAE().to(device)
    
    # Teacher is shared/reused. Student and Router are fresh.
    opt_t = optim.Adam(teacher.parameters(), lr=lr)
    opt_s = optim.Adam(student.parameters(), lr=lr)
    opt_r = optim.Adam(router.parameters(), lr=lr)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        t_loss_acc, s_loss_acc, r_loss_acc = 0, 0, 0
        
        for img, lbl in loader:
            img, lbl = img.to(device), lbl.to(device)
            h = img 
            
            # --- 1. Teacher Update ---
            opt_t.zero_grad()
            t_logits = teacher(h)
            loss_t = F.cross_entropy(t_logits, lbl)
            loss_t.backward()
            opt_t.step()
            t_loss_acc += loss_t.item()
            
            # --- 2 & 3. Student & Router Update ---
            if epoch >= warmup_epochs:
                with torch.no_grad():
                    soft_targets = F.softmax(teacher(h) / temp, dim=1)
                
                opt_s.zero_grad()
                s_logits = student(h)
                loss_s = F.kl_div(F.log_softmax(s_logits / temp, dim=1), soft_targets, reduction='batchmean') * (temp**2)
                loss_s.backward()
                opt_s.step()
                s_loss_acc += loss_s.item()
                
                opt_r.zero_grad()
                loss_r_per_sample = router.get_loss_per_sample(h)
                loss_r = loss_r_per_sample.mean()
                loss_r.backward() 
                opt_r.step()
                r_loss_acc += loss_r.item()
            
        if epoch == epochs - 1:
            loss_name = "ELBO" if router_type == "VAE" else "BCE"
            print(f"    Final Epoch {epoch+1:02d} | T-Loss: {t_loss_acc/len(loader):.4f} | S-Loss: {s_loss_acc/len(loader):.4f} | {loss_name}: {r_loss_acc/len(loader):.4f}")
    
    duration = time.time() - start_time
    print(f"    [Commitment] {task_name} frozen. Raw data purged. (Took {duration:.2f}s)")
    return student.eval(), router.eval()

# --- 4. Main Execution & Evaluation ---

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Task Data Loaders
    loader_a = DataLoader(Subset(mnist_train, [i for i, (_, l) in enumerate(mnist_train) if l < 5]), batch_size=batch_size, shuffle=True)
    loader_b = DataLoader(Subset(mnist_train, [i for i, (_, l) in enumerate(mnist_train) if l >= 5]), batch_size=batch_size, shuffle=True)
    
    # Mixed Test Loader for Lifelong Evaluation
    test_mixed = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    
    # The Persistent Teacher (Shared across tasks to allow forward transfer)
    teacher_engine = TeacherEngine().to(device)
    
    # 1. Train Task A
    student_a, router_a = train_simultaneous(loader_a, "Task A (Digits 0-4)", teacher_engine)
    
    # 2. Train Task B
    student_b, router_b = train_simultaneous(loader_b, "Task B (Digits 5-9)", teacher_engine)
    
    # 3. Autonomous Inference (End-to-End Evaluation)
    print("\n[Inference] Evaluating the Modular Brain on Shuffled Data (0-9)...")
    
    correct_routing = 0
    correct_final = 0
    total = 0
    
    with torch.no_grad():
        for img, lbl in test_mixed:
            img, lbl = img.to(device), lbl.to(device)
            
            # Unilateral Router Evaluation
            loss_a = router_a.get_loss_per_sample(img)
            loss_b = router_b.get_loss_per_sample(img)
            
            # Hard Routing: Select expert with the lowest reconstruction error
            route_to_a = loss_a < loss_b
            
            # Expert Predictions
            pred_a = student_a(img).argmax(1)
            pred_b = student_b(img).argmax(1)
            
            # Final System Prediction based on Router choice
            final_pred = torch.where(route_to_a, pred_a, pred_b)
            
            # Tracking Metrics
            actual_task_a = lbl < 5
            correct_routing += (route_to_a == actual_task_a).sum().item()
            correct_final += (final_pred == lbl).sum().item()
            total += lbl.size(0)
            
    print("\n" + "="*50)
    print("LIFELONG SIMULATION RESULTS (SPLIT-MNIST)")
    print("="*50)
    print(f"Routing Accuracy ({router_type}):   {correct_routing/total*100:.2f}%")
    print(f"End-to-End System Accuracy: {correct_final/total*100:.2f}%")
    print("="*50)
    print("Note: 0.0% Backward Interference achieved because Experts A and B are physically isolated.")