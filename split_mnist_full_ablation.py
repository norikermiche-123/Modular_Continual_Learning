import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import time

# --- 1. System Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
lr = 1e-3
temp = 2.5          

# Robust settings verified in previous run
epochs_base = 25       # Runway for Teacher/Student distillation
warmup_epochs = 3      # Pauses distillation until Teacher is stable
epochs_router = 15     # Epochs to calibrate the routers

vae_latent_dim = 32
tbae_latent_dim = 12

print(f"Full Split-MNIST Ablation Initialized on {device}")
print("Evaluating: [VAE, TB-AE] x [Raw Pixels, Backbone, Student]")

# --- 2. Shared Components ---

class FrozenBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 128), nn.ReLU())
        for param in self.parameters(): param.requires_grad = False
    def forward(self, x): return self.net(x)

class TeacherEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 5))
    def forward(self, x): return self.net(x)

class StudentExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.classifier = nn.Linear(64, 5)
    def forward(self, x): return self.classifier(self.extractor(x))
    def extract_features(self, x): return self.extractor(x) 

# --- 3. Router Architectures ---

class TaskVAE(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, 128), nn.ReLU())
        self.fc_mu = nn.Linear(128, vae_latent_dim)
        self.fc_logvar = nn.Linear(128, vae_latent_dim)
        self.decoder = nn.Sequential(nn.Linear(vae_latent_dim, 128), nn.ReLU(), nn.Linear(128, in_dim))

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        x_flat = x.view(-1, self.in_dim)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
        
    def get_loss(self, x, is_image=False):
        x_flat = x.view(-1, self.in_dim)
        recon, mu, logvar = self.forward(x_flat)
        
        if is_image:
            x_scaled = (x_flat - x_flat.min()) / (x_flat.max() - x_flat.min() + 1e-8)
            recon_scaled = torch.sigmoid(recon)
            recon_loss = F.binary_cross_entropy(recon_scaled, x_scaled, reduction='sum')
        else:
            recon_loss = F.mse_loss(recon, x_flat, reduction='sum')
            
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + KLD

class TaskTBAE(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, tbae_latent_dim), nn.ReLU(),
            nn.Linear(tbae_latent_dim, 128), nn.ReLU(),
            nn.Linear(128, in_dim)
        )
    def get_loss(self, x, is_image=False):
        x_flat = x.view(-1, self.in_dim)
        recon = self.net(x_flat)
        
        if is_image:
            x_scaled = (x_flat - x_flat.min()) / (x_flat.max() - x_flat.min() + 1e-8)
            recon_scaled = torch.sigmoid(recon)
            return F.binary_cross_entropy(recon_scaled, x_scaled, reduction='sum')
        
        return F.mse_loss(recon, x_flat, reduction='sum')

# --- 4. Helper Functions ---

def get_features(img, input_type, backbone, student):
    if input_type == 'x':
        return img
    elif input_type == 'backbone':
        with torch.no_grad(): return backbone(img)
    elif input_type == 'student':
        with torch.no_grad(): return student.extract_features(img)

def train_base_task(loader, task_name):
    """Trains Teacher and Student independently on a specific task."""
    print(f"--- Training Base Models for {task_name} ---")
    teacher = TeacherEngine().to(device)
    student = StudentExpert().to(device)
    opt_t = optim.Adam(teacher.parameters(), lr=lr)
    opt_s = optim.Adam(student.parameters(), lr=lr)
    
    for epoch in range(epochs_base):
        for img, lbl in loader:
            img, lbl = img.to(device), (lbl % 5).to(device)
            
            opt_t.zero_grad()
            t_logits = teacher(img)
            loss_t = F.cross_entropy(t_logits, lbl)
            loss_t.backward()
            opt_t.step()
            
            if epoch >= warmup_epochs:
                with torch.no_grad(): soft_targets = F.softmax(teacher(img) / temp, dim=1)
                opt_s.zero_grad()
                loss_s = F.kl_div(F.log_softmax(student(img)/temp, dim=1), soft_targets, reduction='batchmean') * (temp**2)
                loss_s.backward()
                opt_s.step()
                
    return teacher.eval(), student.eval()

# --- 5. Main Ablation Logic ---

def run_ablation():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_a = DataLoader(Subset(mnist_train, [i for i, (_, l) in enumerate(mnist_train) if l < 5]), batch_size=batch_size, shuffle=True)
    train_b = DataLoader(Subset(mnist_train, [i for i, (_, l) in enumerate(mnist_train) if l >= 5]), batch_size=batch_size, shuffle=True)
    test_mixed = DataLoader(Subset(mnist_test, list(range(1000))), batch_size=batch_size, shuffle=True)
    
    backbone = FrozenBackbone().to(device)
    
    # 1. Train Independent Base Models
    print("\n[Phase 1] Establishing Independent Experts (25 epochs, 3 warmup)...")
    _, student_a = train_base_task(train_a, "Task A (0-4)")
    _, student_b = train_base_task(train_b, "Task B (5-9)")
    
    configs = [
        ('VAE', 'x', 784), ('VAE', 'backbone', 128), ('VAE', 'student', 64),
        ('TBAE', 'x', 784), ('TBAE', 'backbone', 128), ('TBAE', 'student', 64)
    ]
    results = []
    
    # 2. Train and Evaluate Routers
    print("\n[Phase 2] Evaluating 6 Router Configurations...")
    for r_type, i_type, in_dim in configs:
        print(f"  -> Calibrating {r_type} Router on '{i_type}' features...")
        is_img = (i_type == 'x')
        
        RouterClass = TaskVAE if r_type == 'VAE' else TaskTBAE
        router_a = RouterClass(in_dim).to(device)
        router_b = RouterClass(in_dim).to(device)
        opt_ra = optim.Adam(router_a.parameters(), lr=lr)
        opt_rb = optim.Adam(router_b.parameters(), lr=lr)
        
        # Calculate Topological Centering Statistics
        all_h_a = [get_features(img.to(device), i_type, backbone, student_a) for img, _ in train_a]
        mean_a, std_a = torch.cat(all_h_a).mean(dim=0, keepdim=True), torch.cat(all_h_a).std(dim=0, keepdim=True) + 1e-8

        all_h_b = [get_features(img.to(device), i_type, backbone, student_b) for img, _ in train_b]
        mean_b, std_b = torch.cat(all_h_b).mean(dim=0, keepdim=True), torch.cat(all_h_b).std(dim=0, keepdim=True) + 1e-8
        
        # Calibrate Routers independently
        for epoch in range(epochs_router):
            for (img_a, _), (img_b, _) in zip(train_a, train_b):
                h_a = get_features(img_a.to(device), i_type, backbone, student_a)
                h_b = get_features(img_b.to(device), i_type, backbone, student_b)
                
                # Apply centering only to unbounded latents
                if not is_img:
                    h_a = (h_a - mean_a) / std_a
                    h_b = (h_b - mean_b) / std_b
                
                opt_ra.zero_grad(); loss_a = router_a.get_loss(h_a, is_image=is_img); (loss_a/img_a.size(0)).backward(); opt_ra.step()
                opt_rb.zero_grad(); loss_b = router_b.get_loss(h_b, is_image=is_img); (loss_b/img_b.size(0)).backward(); opt_rb.step()
                
        router_a.eval(); router_b.eval()
        
        # Evaluate Routing Accuracy
        correct_routes, total_routes = 0, 0
        with torch.no_grad():
            for img, lbl in test_mixed:
                img, lbl = img.to(device), lbl.to(device)
                
                for i in range(img.size(0)):
                    item_img, item_lbl = img[i:i+1], lbl[i].item()
                    item_h_a = get_features(item_img, i_type, backbone, student_a)
                    item_h_b = get_features(item_img, i_type, backbone, student_b)
                    
                    if not is_img:
                        item_h_a, item_h_b = (item_h_a - mean_a) / std_a, (item_h_b - mean_b) / std_b
                    
                    err_a = router_a.get_loss(item_h_a, is_image=is_img).item()
                    err_b = router_b.get_loss(item_h_b, is_image=is_img).item()
                    
                    predicted_task = 0 if err_a < err_b else 1
                    actual_task = 0 if item_lbl < 5 else 1
                    if predicted_task == actual_task: correct_routes += 1
                    total_routes += 1

        acc = (correct_routes / total_routes) * 100
        results.append((r_type, i_type, acc))

    print("\n" + "="*55)
    print("FINAL SPLIT-MNIST ROUTING ABLATION (INDEPENDENT TASKS)")
    print("="*55)
    print(f"{'Router Loss':<15} | {'Input Features':<15} | {'Routing Acc (%)':<15}")
    print("-" * 55)
    for r_type, i_type, acc in results:
        print(f"{r_type:<15} | {i_type:<15} | {acc:>14.2f}%")
    print("="*55)

if __name__ == "__main__":
    run_ablation()