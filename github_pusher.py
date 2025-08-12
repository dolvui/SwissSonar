import subprocess
from pathlib import Path
import streamlit as st

def push_model_to_github(file_path, commit_msg="Add trained model"):
    repo_url = f"https://{st.secrets['github_token']}@github.com/{st.secrets['repo_name']}.git"
    local_repo = Path("/tmp/repo")

    # Clone
    subprocess.run(["git", "clone", repo_url, local_repo], check=True)

    # Copy file
    dest = local_repo / Path(file_path).name
    subprocess.run(["cp", file_path, dest], check=True)

    # Commit & push
    subprocess.run(["git", "-C", str(local_repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(local_repo), "commit", "-m", commit_msg], check=True)
    subprocess.run(["git", "-C", str(local_repo), "push"], check=True)
