import argparse
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(description="Download model checkpoint from Hugging Face Hub")
parser.add_argument(
    "--repo-id",
    type=str,
    default="zzasdf/viet_iter3_pseudo_label",
    help="Hugging Face repository ID (default: zzasdf/viet_iter3_pseudo_label)"
)
parser.add_argument(
    "--model-dir",
    type=str,
    default="./models/viet_iter3_pseudo_label",
    help="Local directory to save the model (default: ./models/viet_iter3_pseudo_label)"
)

args = parser.parse_args()

print("Downloading checkpoint...")
snapshot_download(
    repo_id=args.repo_id,
    local_dir=args.model_dir,
    local_dir_use_symlinks=False
)
print("Checkpoint saved to {}".format(args.model_dir))