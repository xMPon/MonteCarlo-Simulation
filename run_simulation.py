"""Root CLI script for launching Monte Carlo simulation from project root."""

import os
import sys

if __name__ == "__main__":
    # Forward all arguments to the main CLI in src/main.py
    project_dir = os.path.dirname(os.path.abspath(__file__))
    src_main = os.path.join(project_dir, "src", "main.py")
    if not os.path.exists(src_main):
        print("src/main.py not found.")
        sys.exit(1)
    sys.argv[0] = src_main
    with open(src_main, "rb") as f:
        code = compile(f.read(), src_main, 'exec')
        exec(code, {"__name__": "__main__"})
