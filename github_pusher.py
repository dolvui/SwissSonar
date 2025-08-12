import subprocess
from pathlib import Path
import streamlit as st
import shutil

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
    dest = local_repo / Path(file_path).name
    subprocess.run(["cp", file_path, dest], check=True)

    subprocess.run(["git", "config", "--global", "user.email", "noa@ghidalia.fr"], check=True)
    subprocess.run(["git", "config", "--global", "user.name", "swissSonar"], check=True)

    # Commit & push
    subprocess.run(["git", "pull"], check=True)
    subprocess.run(["git", "-C", str(local_repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(local_repo), "commit", "-m", commit_msg], check=True)
    subprocess.run(["git", "-C", str(local_repo), "push"], check=True)
