"""
Fix imports for database adapter to work with your algorithm classes.
Enhanced version for GUI compatibility.
"""

import sys
import os
from pathlib import Path


def fix_algorithm_imports():
    """Add correct paths to your algorithm modules."""

    # Get project root (adjust if needed)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent

    # Add your src directories
    src_paths = [
        project_root / "src" / "utils",
        project_root / "src" / "models",
        project_root / "src" / "algorithms"
    ]

    for path in src_paths:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
                print(f"Added to path: {path}")
        else:
            print(f"Path not found: {path}")

    # Try importing core classes
    try:
        from data_loader import DataLoader
        from problem import CourseSchedulingProblem
        print("✓ Successfully imported core classes")

        # Try importing algorithms
        algorithm_imports = []
        try:
            from greedy import GreedyAlgorithm
            algorithm_imports.append("GreedyAlgorithm")
        except ImportError:
            print("! Could not import GreedyAlgorithm")

        try:
            from genetic import GeneticAlgorithm
            algorithm_imports.append("GeneticAlgorithm")
        except ImportError:
            print("! Could not import GeneticAlgorithm")

        try:
            from simulated_annealing import SimulatedAnnealingAlgorithm
            algorithm_imports.append("SimulatedAnnealingAlgorithm")
        except ImportError:
            print("! Could not import SimulatedAnnealingAlgorithm")

        try:
            from tabu_search import TabuSearchAlgorithm
            algorithm_imports.append("TabuSearchAlgorithm")
        except ImportError:
            print("! Could not import TabuSearchAlgorithm")

        try:
            from vns import VariableNeighborhoodSearch
            algorithm_imports.append("VariableNeighborhoodSearch")
        except ImportError:
            print("! Could not import VariableNeighborhoodSearch")

        if algorithm_imports:
            print(f"✓ Successfully imported algorithms: {', '.join(algorithm_imports)}")
        else:
            print("! No algorithms could be imported")

        return True

    except ImportError as e:
        print(f"✗ Still cannot import core classes: {e}")
        print("Available paths:")
        for p in sys.path:
            print(f"  {p}")
        return False


def get_algorithm_path():
    """Get the path to algorithms directory."""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    return project_root / "src" / "algorithms"


def test_algorithm_import(algorithm_name):
    """Test if specific algorithm can be imported."""
    try:
        algorithms_path = get_algorithm_path()
        if str(algorithms_path) not in sys.path:
            sys.path.append(str(algorithms_path))

        if algorithm_name == "GreedyAlgorithm":
            from greedy import GreedyAlgorithm
            return True
        elif algorithm_name == "GeneticAlgorithm":
            from genetic import GeneticAlgorithm
            return True
        elif algorithm_name == "SimulatedAnnealingAlgorithm":
            from simulated_annealing import SimulatedAnnealingAlgorithm
            return True
        elif algorithm_name == "TabuSearchAlgorithm":
            from tabu_search import TabuSearchAlgorithm
            return True
        elif algorithm_name == "VariableNeighborhoodSearch":
            from vns import VariableNeighborhoodSearch
            return True
        else:
            return False
    except ImportError:
        return False


if __name__ == "__main__":
    print("Testing algorithm imports...")
    success = fix_algorithm_imports()

    if success:
        print("\n✓ Import fixing completed successfully!")
    else:
        print("\n✗ Import fixing failed!")

    # Test individual algorithms
    algorithms = ["GreedyAlgorithm", "GeneticAlgorithm", "SimulatedAnnealingAlgorithm",
                  "TabuSearchAlgorithm", "VariableNeighborhoodSearch"]

    print("\nTesting individual algorithm imports:")
    for alg in algorithms:
        result = test_algorithm_import(alg)
        status = "✓" if result else "✗"
        print(f"  {status} {alg}")