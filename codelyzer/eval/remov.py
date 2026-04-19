import os
from pathlib import Path
p = Path("/Users/vijaymurugan/Desktop/projects/bill-split-main/").resolve()
print("exists:", p.exists())
print("is_dir:", p.is_dir())
print("resolved:", p)
from git import Repo
Repo(p)