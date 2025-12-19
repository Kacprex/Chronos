import subprocess
import os
import sys

ZIP_NAME = "latest.zip"


def run(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        print("❌ Command failed:", " ".join(cmd))
        print(result.stderr)
        sys.exit(1)


def main():
    # Ensure we're in a git repo
    run(["git", "rev-parse", "--is-inside-work-tree"])

    # Remove existing zip
    if os.path.exists(ZIP_NAME):
        os.remove(ZIP_NAME)

    # Create archive from HEAD (respects .gitignore automatically)
    run([
        "git", "archive",
        "--format=zip",
        f"--output={ZIP_NAME}",
        "HEAD"
    ])

    print(f"✔ Created {ZIP_NAME} (tracked files only)")


if __name__ == "__main__":
    main()
