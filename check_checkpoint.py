# Script check if a checkpoint is causal or non-causal (streaming or non-streaming)
import torch

checkpoint_path = "/home/ubuntu/HoangLT19/VietASR/models/viet_iter3_pseudo_label/exp/epoch-12-avg-8.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

keys = list(checkpoint["model"].keys())
causal_keys = [k for k in keys if "causal_conv" in k]
non_causal_keys = [k for k in keys if "depthwise_conv.weight" in k]

print(f"Total keys: {len(keys)}")
print(f"Causal keys found: {len(causal_keys)}")
print(f"Non-Causal keys found: {len(non_causal_keys)}")

if len(causal_keys) > 0:
    print("This checkpoint appears to be CAUSAL.")
elif len(non_causal_keys) > 0:
    print("This checkpoint appears to be NON-CAUSAL.")
else:
    print("Could not determine causality from keys.")
