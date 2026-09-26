"""PyInstaller entry point (the package uses relative imports, so it can't be
the entry script itself)."""
from visualizer.main import main

if __name__ == "__main__":
    main()
