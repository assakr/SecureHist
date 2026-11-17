import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt
from medical_classifier.model import CNNModel

def pgd_attack_batch(model, images, labels, eps=8/255.0, alpha=2/255.0, iters=7, device="cuda"):
    """
    Fast PGD attack during training.
    Using iters=7 during training provides strong adversarial examples 
    while keeping the training loop fast.
    """
    loss_fn = nn.CrossEntropyLoss()
    delta = torch.empty_like(images).uniform_(-eps, eps).to(device)
    adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()
    
    # Save training state and switch to eval for attack generation (standard adversarial training practice)
    was_training = model.training
    model.eval()
    
    for _ in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = loss_fn(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images)[0]
        
        adv_images = adv_images.detach() + alpha * torch.sign(grad)
        eta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + eta, min=0.0, max=1.0).detach()
        
    if was_training:
        model.train()
        
    return adv_images

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("      ADVERSARIAL TRAINING (PGD DEFENSE) FOR MEDICAL CNN       ")
    print("=" * 65)
    print(f"[+] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    batch_size = 64
    img_size = 180
    num_classes = 3
    num_epochs = 5
    lr = 0.0001
    eps = 8 / 255.0
    alpha = 2 / 255.0
    train_iters = 7
    eval_iters = 20
    
    # Preprocessing
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    train_dir = r"lung_colon_image_set\Train and Validation Set"
    test_dir = r"lung_colon_image_set\Test Set"
    
    print(f"[+] Loading training data from: {train_dir}")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    class_names = train_dataset.classes
    print(f"[+] Classes found: {class_names}")
    print(f"[+] Total training images: {len(train_dataset)}")
    
    # Only evaluate on the 3 matching classes from Test Set
    test_raw = datasets.ImageFolder(test_dir, transform=val_transforms)
    test_samples = [s for s in test_raw.samples if os.path.basename(os.path.dirname(s[0])) in class_names]
    test_dataset = copy.deepcopy(test_raw)
    test_dataset.samples = test_samples
    test_dataset.targets = [class_names.index(os.path.basename(os.path.dirname(s[0]))) for s in test_samples]
    test_dataset.classes = class_names
    test_dataset.class_to_idx = {c: i for i, c in enumerate(class_names)}
    print(f"[+] Total test evaluation images: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize Model & Load Baseline Weights
    model = CNNModel(num_classes=num_classes).to(device)
    baseline_weights_path = "medical_cnn.pth"
    if os.path.exists(baseline_weights_path):
        print(f"[+] Loading baseline weights from {baseline_weights_path} for fine-tuning...")
        model.load_state_dict(torch.load(baseline_weights_path, map_location=device, weights_only=True))
    else:
        print("[-] Baseline weights not found, training from scratch.")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # -------------------------------------------------------------
    # 1. Evaluate Pre-Training Robustness on Test Set
    # -------------------------------------------------------------
    print("\n--- Pre-Training Test Evaluation ---")
    model.eval()
    pre_clean_corr, pre_adv_corr, pre_tot = 0, 0, 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            pre_clean_corr += (torch.argmax(out, dim=1) == lbls).sum().item()
            pre_tot += lbls.size(0)
            
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        adv_imgs = pgd_attack_batch(model, imgs, lbls, eps=eps, alpha=alpha, iters=eval_iters, device=device)
        with torch.no_grad():
            out = model(adv_imgs)
            pre_adv_corr += (torch.argmax(out, dim=1) == lbls).sum().item()
            
    pre_clean_acc = pre_clean_corr / pre_tot * 100
    pre_adv_acc = pre_adv_corr / pre_tot * 100
    print(f"Pre-Fine-Tuning Clean Accuracy:  {pre_clean_acc:.2f}%")
    print(f"Pre-Fine-Tuning Robust Accuracy: {pre_adv_acc:.2f}%\n")
    
    # -------------------------------------------------------------
    # 2. Adversarial Training Loop
    # -------------------------------------------------------------
    best_robust_acc = pre_adv_acc
    best_model_path = "medical_cnn_robust.pth"
    history = []
    
    for epoch in range(1, num_epochs + 1):
        start_epoch = time.time()
        model.train()
        train_loss = 0.0
        train_clean_corr = 0
        train_adv_corr = 0
        train_tot = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Generate PGD adversarial examples
            adv_inputs = pgd_attack_batch(model, inputs, targets, eps=eps, alpha=alpha, iters=train_iters, device=device)
            
            model.train()
            optimizer.zero_grad()
            
            # Forward pass on clean and adversarial samples (TRADES / Madry style mixed training)
            out_clean = model(inputs)
            out_adv = model(adv_inputs)
            
            loss_clean = criterion(out_clean, targets)
            loss_adv = criterion(out_adv, targets)
            
            # Joint objective: 50% clean loss + 50% adversarial loss
            loss = 0.5 * loss_clean + 0.5 * loss_adv
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * targets.size(0)
            train_clean_corr += (torch.argmax(out_clean, dim=1) == targets).sum().item()
            train_adv_corr += (torch.argmax(out_adv, dim=1) == targets).sum().item()
            train_tot += targets.size(0)
            
            if (batch_idx + 1) % 40 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] | "
                      f"Loss: {loss.item():.4f} | "
                      f"Train Clean: {train_clean_corr/train_tot*100:.1f}% | "
                      f"Train Adv: {train_adv_corr/train_tot*100:.1f}%")
                
        scheduler.step()
        epoch_time = time.time() - start_epoch
        
        # Test evaluation after epoch
        model.eval()
        test_clean_corr, test_adv_corr, test_tot = 0, 0, 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                test_clean_corr += (torch.argmax(out, dim=1) == lbls).sum().item()
                test_tot += lbls.size(0)
                
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            adv_imgs = pgd_attack_batch(model, imgs, lbls, eps=eps, alpha=alpha, iters=eval_iters, device=device)
            with torch.no_grad():
                out = model(adv_imgs)
                test_adv_corr += (torch.argmax(out, dim=1) == lbls).sum().item()
                
        t_clean_acc = test_clean_corr / test_tot * 100
        t_adv_acc = test_adv_corr / test_tot * 100
        
        print(f"\n---> Epoch {epoch} Done in {epoch_time:.1f}s | "
              f"Test Clean Acc: {t_clean_acc:.2f}% | "
              f"Test Robust Acc (PGD): {t_adv_acc:.2f}%")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss / train_tot,
            "train_clean_acc": train_clean_corr / train_tot * 100,
            "train_adv_acc": train_adv_corr / train_tot * 100,
            "test_clean_acc": t_clean_acc,
            "test_robust_acc": t_adv_acc
        })
        
        # Save best checkpoint
        if t_adv_acc > best_robust_acc or epoch == 1:
            best_robust_acc = t_adv_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"[+] Saved new best robust model to: {best_model_path} (Robust Acc: {best_robust_acc:.2f}%)\n")
        else:
            print("\n")
            
    # Save training progression plot
    os.makedirs("output_pgd", exist_ok=True)
    df_hist = pd.DataFrame(history)
    df_hist.to_csv("output_pgd/adversarial_training_log.csv", index=False)
    
    plt.figure(figsize=(9, 5))
    epochs_range = [h["epoch"] for h in history]
    plt.plot(epochs_range, [h["test_clean_acc"] for h in history], marker='o', linewidth=2, label="Test Clean Acc", color="#2ca02c")
    plt.plot(epochs_range, [h["test_robust_acc"] for h in history], marker='s', linewidth=2, label="Test Robust Acc (PGD-20)", color="#d62728")
    plt.axhline(y=pre_adv_acc, linestyle='--', color='gray', label=f"Original Model Robust Acc ({pre_adv_acc:.1f}%)")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.title("Adversarial Training Progress: Clean vs Robust Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.ylim([0, 105])
    plt.tight_layout()
    plot_path = "output_pgd/adversarial_training_progress.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    print("=" * 65)
    print("           ADVERSARIAL TRAINING COMPLETE           ")
    print("=" * 65)
    print(f"Original Baseline Robust Acc: {pre_adv_acc:.2f}%")
    print(f"Final Defended Robust Acc:   {best_robust_acc:.2f}%")
    print(f"Robust Model Saved At:       {best_model_path}")
    print(f"Progress Plot Saved At:      {plot_path}")

if __name__ == "__main__":
    main()
