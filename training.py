import os, time, argparse, numpy as np
import torch, torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from load_data import get_loaders
from model import SmallCNN

def train_one_epoch(model, loader, opt, device):
    model.train()
    running, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.05)
        loss.backward(); opt.step()
        running += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return running/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    running, correct, total = 0.0, 0, 0
    ys, ps = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        running += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        ys.append(y.cpu().numpy()); ps.append(pred.cpu().numpy())
    return running/total, correct/total, np.concatenate(ys), np.concatenate(ps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader, classes = get_loaders(
        dataset=args.dataset, bs_train=args.bs, bs_test=256, num_workers=args.num_workers
    )

    model = SmallCNN(num_classes=len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", f"{args.dataset}_cnn.pt")

    best_acc = 0.0
    for e in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device)
        te_loss, te_acc, y_true, y_pred = evaluate(model, test_loader, device)
        print(f"Epoch {e:02d} | train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"test {te_loss:.4f}/{te_acc:.4f} | {time.time()-t0:.1f}s")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({"model_state": model.state_dict(), "classes": classes}, save_path)

    # Final report on best model
    ckpt = torch.load(save_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    classes = [str(c) for c in ckpt["classes"]]
    _, best_acc_final, y_true, y_pred = evaluate(model, test_loader, device)

    print("\nBest test accuracy:", best_acc)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
