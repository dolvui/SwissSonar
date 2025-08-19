import subprocess
from pathlib import Path
import streamlit as st
import shutil
import os
from datetime import datetime

def push_model_to_github(files_path, commit_msg="Add trained model",token=None,repo=None):
    if token is None:
        token = st.secrets['github']['github_token']
    if repo is None:
        repo = st.secrets['github']['repo_name']
    repo_url = f"https://{token}@github.com/{repo}.git"
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

    for path in files_path:
        dest = models_dir / Path(path).name
        subprocess.run(["cp", path, dest], check=True)

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
    bench_path = models_dir / model_name.replace(".pt", "_norms.npy")

    # Check if model exists
    if model_path.exists():
        os.remove(model_path)
        os.remove(bench_path)
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


def request_training(id,training_crypto):
    repo_url = f"https://{st.secrets['github']['github_token']}@github.com/{st.secrets['github']['repo_name']}.git"
    local_repo = Path("/tmp/repo")
    if local_repo.exists():
        shutil.rmtree(local_repo)
    subprocess.run(["git", "clone", repo_url, local_repo], check=True)

    jobs_dir = local_repo / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    job_file = jobs_dir / f"train_{id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    job_file.write_text(f"{id}\n{":".join(training_crypto)}")


    subprocess.run(["git", "config", "--global", "user.email", "noa@ghidalia.fr"], check=True)
    subprocess.run(["git", "config", "--global", "user.name", "swissSonar"], check=True)

    subprocess.run(["git", "-C", str(local_repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(local_repo), "commit", "-m", f"Request training for model id : {id}"], check=True)
    subprocess.run(["git", "-C", str(local_repo), "push"], check=True)

def push_db_to_github(db_path="/tmp/models.db", commit_msg="Update models.db", token=None, repo=None):
    import hashlib, os, subprocess, shutil
    from pathlib import Path
    import streamlit as st

    if token is None:
        token = st.secrets['github']['github_token']
    if repo is None:
        repo = st.secrets['github']['repo_name']

    repo_url = f"https://{token}@github.com/{repo}.git"
    local_repo = Path("/tmp/repo")

    # Clean old clone
    if local_repo.exists():
        shutil.rmtree(local_repo)

    subprocess.run(["git", "clone", repo_url, str(local_repo)], check=True)

    dest = local_repo / Path(db_path).name
    shutil.copy2(db_path, dest)

    subprocess.run(["git", "-C", str(local_repo), "config", "user.email", "noa@ghidalia.fr"], check=True)
    subprocess.run(["git", "-C", str(local_repo), "config", "user.name", "swissSonar"], check=True)

    # Stage and commit
    subprocess.run(["git", "-C", str(local_repo), "add", "-f", str(dest)], check=True)
    result = subprocess.run(["git", "-C", str(local_repo), "status", "--porcelain"], capture_output=True, text=True)

    if result.stdout.strip():
        subprocess.run(["git", "-C", str(local_repo), "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "-C", str(local_repo), "push"], check=True)
        st.success("✅ models.db updated and pushed")
    else:
        st.info("ℹ️ models.db already up-to-date, no commit made")
