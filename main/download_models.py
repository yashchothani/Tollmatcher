from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Infomaniak-AI/onnx-opus-mt-fr-it",
    cache_dir="/home/tollmatcher2/.cache/huggingface",
    local_dir_use_symlinks=False
)
