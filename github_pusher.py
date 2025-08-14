import subprocess
from pathlib import Path
import streamlit as st
import shutil
import os

def push_model_to_github(file_path, commit_msg="Add trained model"):
    repo_url = f"https://{st.secrets['github']['github_token']}@github.com/{st.secrets['github']['repo_name']}.git"
    local_repo = Path("/tmp/repo")
    if local_repo.exists():
        shutil.rmtree(local_repo)
    # Clone
    try:
        subprocess.run(["git", "clone", repo_url, local_repo], check=True)
    except:
        pass
    # Copy file
    models_dir = local_repo / "models"
    #models_dir.mkdir(parents=True, exist_ok=True)

    dest = models_dir / Path(file_path).name
    subprocess.run(["cp", file_path, dest], check=True)

    subprocess.run(["git", "config", "--global", "user.email", "noa@ghidalia.fr"], check=True)
    subprocess.run(["git", "config", "--global", "user.name", "swissSonar"], check=True)

    # Commit & push
    subprocess.run(["git", "pull"])
    subprocess.run(["git", "-C", str(local_repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(local_repo), "commit", "-m", commit_msg], check=True)
    subprocess.run(["git", "-C", str(local_repo), "push"], check=True)

def delete_model_from_github(model_name, commit_msg="Delete model"):
    repo_url = f"https://{st.secrets['github']['github_token']}@github.com/{st.secrets['github']['repo_name']}.git"
    local_repo = Path("/tmp/repo")

    # Remove any previous clone
    if local_repo.exists():
        shutil.rmtree(local_repo)

    # Clone repo
    subprocess.run(["git", "clone", repo_url, local_repo], check=True)

    # Path to models dir in repo
    models_dir = local_repo / "models"
    model_path = models_dir / model_name

    # Check if model exists
    if model_path.exists():
        os.remove(model_path)
        print(f"Deleted {model_name} from repo")
    else:
        raise FileNotFoundError(f"Model '{model_name}' not found in repository models directory.")

    # Git config
    subprocess.run(["git", "config", "--global", "user.email", "noa@ghidalia.fr"], check=True)
    subprocess.run(["git", "config", "--global", "user.name", "swissSonar"], check=True)

    # Commit & push
    subprocess.run(["git", "-C", str(local_repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(local_repo), "commit", "-m", commit_msg], check=True)
    subprocess.run(["git", "-C", str(local_repo), "push"], check=True)