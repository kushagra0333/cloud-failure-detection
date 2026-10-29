"""Inspect saved checkpoints in models/checkpoints_phase7i/."""
import torch
from pathlib import Path

ckpt_dir = Path("models/checkpoints_phase7i")
ckpts = {
    "condA": ckpt_dir / "transformer_condA_unweighted.pt",
    "condB": ckpt_dir / "transformer_condB_moderate.pt",
    "condC": ckpt_dir / "transformer_condC_stronger.pt",
    "condD": ckpt_dir / "transformer_condD_higher.pt"
}

for name, path in ckpts.items():
    data = torch.load(path, map_location="cpu")
    print(f"\nCheckpoint {name} ({path.name}):")
    print("  Keys:", list(data.keys()))
    print("  Saved epoch:", data.get("epoch"))
    print("  Saved val_loss:", data.get("val_loss"))
    state = data["model_state_dict"]
    print("  Number of weight tensors:", len(state))
    # Print sample weight norm or summary
    first_key = list(state.keys())[0]
    last_key = list(state.keys())[-1]
    print(f"  {first_key} shape: {state[first_key].shape}, norm: {state[first_key].norm().item():.4f}")
    print(f"  {last_key} shape: {state[last_key].shape}, norm: {state[last_key].norm().item():.4f}")

# Check if model weights are identical or distinct between conditions
state_A = torch.load(ckpts["condA"], map_location="cpu")["model_state_dict"]
state_B = torch.load(ckpts["condB"], map_location="cpu")["model_state_dict"]
state_C = torch.load(ckpts["condC"], map_location="cpu")["model_state_dict"]
state_D = torch.load(ckpts["condD"], map_location="cpu")["model_state_dict"]

def are_states_equal(s1, s2):
    for k in s1:
        if not torch.equal(s1[k], s2[k]):
            return False
    return True

print("\nWeight equality checks across saved checkpoints:")
print("  Cond A == Cond B?", are_states_equal(state_A, state_B))
print("  Cond A == Cond C?", are_states_equal(state_A, state_C))
print("  Cond A == Cond D?", are_states_equal(state_A, state_D))
print("  Cond B == Cond C?", are_states_equal(state_B, state_C))
print("  Cond C == Cond D?", are_states_equal(state_C, state_D))
