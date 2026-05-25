import argparse
import subprocess
import sys
import os

EMONET_REPO_URL = "https://github.com/ecco-laboratory/emonet-pytorch.git"
DEFAULT_TARGET  = "emonet-pytorch"
N_CLASSES       = 20


def run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def clone_repo(target: str) -> None:
    run(["git", "config", "--global", "--add", "safe.directory", target])
    if not os.path.isdir(os.path.join(target, ".git")):
        print(f"Cloning {EMONET_REPO_URL} → {target}")
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        run(["git", "clone", EMONET_REPO_URL, target])
    else:
        print(f"Repo already exists at {target} — skipping clone.")
        
def check_deps() -> None:
    print("\nChecking dependencies...")
    for pkg in ("torch", "torchvision"):
        try:
            m = __import__(pkg)
            print(f"  {pkg} {m.__version__}  ✓")
        except ImportError:
            print(f"  {pkg} NOT FOUND — installing...")
            run([sys.executable, "-m", "pip", "install", pkg, "--quiet"])


def verify(target: str) -> None:
    import torch
    import numpy as np
    from PIL import Image
    import torchvision.transforms as T

    print(f"\nVerifying EmoNet from {target}...")

    if target not in sys.path:
        sys.path.insert(0, target)

    try:
        from models import EmoNet  
    except ImportError as e:
        print(f"\nERROR: could not import from {target}/models.py")
        print(f"  {e}")
        print("  Check the repo cloned correctly and contains models.py")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}")

    print("  Loading EmoNet + downloading weights from OSF (first run only)...")
    model = EmoNet()
    model.load_state_dict_from_web()   
    model.eval()
    model.to(device)
    print("  Model loaded  ✓")

    transform = T.Compose([
        T.Resize((227, 227)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dummy = Image.fromarray(
        (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    )
    x = transform(dummy).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (1, N_CLASSES), \
        f"Bad output shape: {logits.shape}, expected (1, {N_CLASSES})"
    probs = torch.softmax(logits, dim=-1)
    assert abs(probs.sum().item() - 1.0) < 1e-4
    print(f"  Output shape : {tuple(logits.shape)}  ✓")
    print(f"  Softmax sum  : {probs.sum().item():.6f}  ✓")
    print(f"  Top category : {probs.argmax().item()} "
          f"(p={probs.max().item():.3f})")

    try:
        from cloned.rewards.blackbox_rewards import PupilEmoNetReward
        reward = PupilEmoNetReward(
            emonet_repo_path=target, device=device,
            w_luminance=0.55, w_arousal=0.30, w_contrast=0.15,
            noise_std=0.0,
        )
        scores = reward.score([dummy, dummy])
        assert len(scores) == 2 and all(0.0 <= s <= 1.0 for s in scores)
        print(f"  PupilEmoNetReward scores : {[f'{s:.4f}' for s in scores]}  ✓")
    except ImportError:
        print("  (PupilEmoNetReward check skipped — run from project root)")

    print(f"\n✓ All checks passed.")
    print(f"\nSet in main_pupil_proxy.py:")
    print(f'  EMONET_REPO_PATH = "{target}"')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",      default=DEFAULT_TARGET)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    print("=" * 55)
    print("  EmoNet setup (Kragel 2019 / ecco-laboratory)")
    print("=" * 55)

    check_deps()

    if not args.verify_only:
        clone_repo(args.target)
    else:
        if not os.path.isdir(args.target):
            print(f"ERROR: {args.target} does not exist.")
            sys.exit(1)
        print(f"\nSkipping clone — using {args.target}")

    verify(args.target)


if __name__ == "__main__":
    main()