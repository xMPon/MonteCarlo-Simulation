"""Root CLI script for launching Monte Carlo simulation from project root."""

import os
import sys

if __name__ == "__main__":
    # Find the main.py script in the project root
    project_dir = os.path.dirname(os.path.abspath(__file__))
    main_py = os.path.join(project_dir, "main.py")
    if not os.path.exists(main_py):
        print("main.py not found.")
        sys.exit(1)
    # Set sys.argv[0] to main.py so argparse and error messages are correct
    sys.argv[0] = main_py
    # Open and execute main.py as the main module
    with open(main_py, "rb") as f:
        code = compile(f.read(), main_py, 'exec')
        exec(code, {"__name__": "__main__"})
