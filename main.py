#!/usr/bin/env python3
"""
Course Scheduling Optimization System - Main Entry Point

This is the main orchestrator for the course scheduling optimization system.
It provides a unified interface for running experiments, analyzing results,
and managing the entire research workflow.

Usage:
    python main.py                              # Interactive mode
    python main.py --mode benchmark             # Run benchmarks
    python main.py --mode analyze              # Analyze results
    python main.py --mode single --algorithm greedy --instance small_01
    python main.py --help                      # Show all options
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
import time
from datetime import datetime

# Add src directories to Python path
PROJECT_ROOT = Path(__file__).parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.extend([
    str(SRC_ROOT / "utils"),
    str(SRC_ROOT / "models"),
    str(SRC_ROOT / "algorithms"),
    str(SRC_ROOT / "experiments")
])

# Import system modules
try:
    from experiments.benchmark import BatchRunner
    from utils.metrics import ResultsAnalyzer
    from utils.data_loader import load_instance
    from models.problem import CourseSchedulingProblem

    # Import algorithms
    from algorithms.greedy import GreedyAlgorithm, create_greedy_variants
    from algorithms.genetic import GeneticAlgorithm, create_genetic_variants
    # from algorithms.simulated_annealing import SimulatedAnnealingAlgorithm, create_sa_variants
    # from algorithms.tabu_search import TabuSearchAlgorithm, create_tabu_variants
    # from algorithms.vns import VariableNeighborhoodSearch, create_vns_variants

    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"⚠️  Import Error: {e}")
    print("Some modules may not be available. Continuing with limited functionality...")
    IMPORTS_SUCCESS = False


class CourseSchedulingSystem:
    """
    Main system orchestrator for course scheduling optimization.

    Provides unified interface for all system operations including
    benchmarking, analysis, visualization, and single algorithm runs.
    """

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the system."""
        self.project_root = project_root or PROJECT_ROOT
        self.data_dir = self.project_root / "data"
        self.instances_dir = self.data_dir / "instances"
        self.results_dir = self.data_dir / "results"
        self.config_dir = self.project_root / "config"

        # Ensure directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.batch_runner = None
        self.results_analyzer = None

        if IMPORTS_SUCCESS:
            self.batch_runner = BatchRunner(str(self.project_root))
            self.results_analyzer = ResultsAnalyzer(str(self.results_dir))

        # System information
        self.available_algorithms = {
            'greedy': (GreedyAlgorithm, create_greedy_variants),
            'genetic': (GeneticAlgorithm, create_genetic_variants),
            # 'sa': (SimulatedAnnealingAlgorithm, create_sa_variants),
            # 'tabu': (TabuSearchAlgorithm, create_tabu_variants),
            # 'vns': (VariableNeighborhoodSearch, create_vns_variants)
        } if IMPORTS_SUCCESS else {}

        print(f"🚀 Course Scheduling System initialized")
        print(f"   Project root: {self.project_root}")
        print(
            f"   Available algorithms: {list(self.available_algorithms.keys()) if IMPORTS_SUCCESS else 'None (import errors)'}")

    def run_benchmark(self, profile: str = "quick", instances: str = None,
                      algorithms: str = None, runs: int = None, verbose: bool = False) -> bool:
        """
        Run benchmark experiments.

        Args:
            profile: Benchmark profile (quick, comparison, full)
            instances: Instance pattern override
            algorithms: Algorithm pattern override
            runs: Number of runs override
            verbose: Verbose output

        Returns:
            True if successful
        """
        if not self.batch_runner:
            print("❌ Batch runner not available due to import errors")
            return False

        print(f"🔬 Running benchmark profile: {profile}")

        # Define benchmark profiles
        profiles = {
            "quick": {
                "instances": instances or "small/small_01",
                "algorithms": algorithms or "greedy",
                "runs": runs or 1,
                "description": "Quick test with single instance and algorithm"
            },
            "comparison": {
                "instances": instances or "small/*",
                "algorithms": algorithms or ["greedy", "genetic", "sa"],
                "runs": runs or 1,
                "description": "Compare main algorithms on small instances"
            },
            "variants": {
                "instances": instances or "*/*",
                "algorithms": algorithms or ["greedy_variants", "genetic_variants"],
                "runs": runs or 3,
                "description": "Test algorithm variants"
            },
            "full": {
                "instances": instances or "*/*",
                # "algorithms": algorithms or "all",
                "algorithms": algorithms or  ["greedy_variants", "genetic_variants"],
                "runs": runs or 3,
                "description": "Comprehensive benchmark (WARNING: This will take a long time!)"
            }
        }

        if profile not in profiles:
            print(f"❌ Unknown profile: {profile}")
            print(f"Available profiles: {list(profiles.keys())}")
            return False

        config = profiles[profile]
        print(f"📋 {config['description']}")

        # Confirm if running full benchmark
        if profile == "full":
            response = input("This will run ALL algorithms on ALL instances with multiple runs. Continue? (y/N): ")
            if response.lower() != 'y':
                print("Benchmark cancelled.")
                return False

        # Run benchmark
        try:
            start_time = time.time()
            results = self.batch_runner.run_batch(
                instances=config["instances"],
                algorithms=config["algorithms"],
                runs_per_combination=config["runs"],
                verbose=verbose
            )
            duration = time.time() - start_time

            successful = sum(1 for r in results if r.get('success', False))
            total = len(results)

            print(f"✅ Benchmark completed in {duration:.1f}s")
            print(f"   Results: {successful}/{total} successful")
            print(f"   Results saved to: {self.results_dir}")

            return successful > 0

        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return False

    def run_single_experiment(self, algorithm: str, instance: str,
                              variant: str = None, verbose: bool = True) -> bool:
        """
        Run single algorithm on single instance.

        Args:
            algorithm: Algorithm name
            instance: Instance name or path
            variant: Specific algorithm variant
            verbose: Verbose output

        Returns:
            True if successful
        """
        if not self.batch_runner:
            print("❌ Cannot run experiment due to import errors")
            return False

        print(f"🧪 Running single experiment: {algorithm} on {instance}")

        # Determine algorithm specification
        if variant:
            alg_spec = f"{algorithm}_{variant}"
        else:
            alg_spec = algorithm

        try:
            results = self.batch_runner.run_batch(
                instances=instance,
                algorithms=alg_spec,
                runs_per_combination=1,
                verbose=verbose
            )

            if results and results[0].get('success'):
                result = results[0]
                obj_val = result.get('objective_value', 'N/A')
                feasible = result.get('is_feasible', False)
                time_taken = result.get('execution_time', 0)

                print(f"✅ Experiment successful!")
                print(f"   Objective value: {obj_val}")
                print(f"   Feasible: {feasible}")
                print(f"   Execution time: {time_taken:.3f}s")
                return True
            else:
                print(f"❌ Experiment failed")
                return False

        except Exception as e:
            print(f"❌ Experiment error: {e}")
            return False

    def analyze_results(self, focus: str = "overview", export_csv: bool = True) -> bool:
        """
        Analyze experiment results.

        Args:
            focus: Analysis focus (overview, comparison, statistical)
            export_csv: Whether to export CSV files

        Returns:
            True if successful
        """
        if not self.results_analyzer:
            print("❌ Results analyzer not available due to import errors")
            return False

        print(f"📊 Analyzing results with focus: {focus}")

        try:
            # Load results
            self.results_analyzer.load_results()

            if not self.results_analyzer.raw_results:
                print("❌ No results found to analyze")
                print(f"   Check results directory: {self.results_dir}")
                return False

            # Generate analysis based on focus
            if focus == "overview":
                self.results_analyzer.print_summary_report()

            elif focus == "comparison":
                # Get available algorithms
                algorithms = list(set(alg for (alg, inst), _ in self.results_analyzer.algorithm_performances.items()))

                if len(algorithms) >= 2:
                    print(f"\n📈 Statistical Comparisons:")
                    print("=" * 60)

                    # Compare all pairs
                    for i in range(len(algorithms)):
                        for j in range(i + 1, len(algorithms)):
                            alg_a, alg_b = algorithms[i], algorithms[j]
                            comparisons = self.results_analyzer.compare_algorithms(alg_a, alg_b)

                            if comparisons:
                                print(f"\n{alg_a} vs {alg_b}:")
                                for comp in comparisons:
                                    sig = "***" if comp.p_value < 0.001 else "**" if comp.p_value < 0.01 else "*" if comp.p_value < 0.05 else ""
                                    print(f"  {comp.instance_name}: {comp.winner} wins (p={comp.p_value:.4f}){sig}")
                else:
                    print(f"❌ Need at least 2 algorithms for comparison, found: {len(algorithms)}")

            elif focus == "statistical":
                # Detailed statistical analysis
                ranking = self.results_analyzer.generate_performance_ranking()

                print(f"\n🏆 Algorithm Ranking:")
                print("=" * 60)
                for alg in ranking:
                    print(f"{alg['rank']}. {alg['algorithm_name']}")
                    print(f"   Avg Objective: {alg['avg_objective']:.2f} ± {alg['std_objective']:.2f}")
                    print(f"   Success Rate: {alg['avg_success_rate']:.1%}")
                    print(f"   Avg Time: {alg['avg_time']:.3f}s")
                    print()

            # Export CSV files
            if export_csv:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                summary_file = self.results_dir / f"analysis_summary_{timestamp}.csv"
                raw_file = self.results_dir / f"analysis_raw_{timestamp}.csv"

                self.results_analyzer.export_to_csv(str(summary_file))
                self.results_analyzer.export_to_csv(str(raw_file), include_raw_data=True)

                print(f"📁 Analysis exported to:")
                print(f"   Summary: {summary_file}")
                print(f"   Raw data: {raw_file}")

            return True

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return False

    def list_available_resources(self) -> None:
        """List available instances, algorithms, and recent results."""
        print(f"\n📚 Available Resources:")
        print("=" * 50)

        # List instances
        print(f"🗂️  Instances:")
        if self.instances_dir.exists():
            for category_dir in sorted(self.instances_dir.iterdir()):
                if category_dir.is_dir():
                    instances = [inst.name for inst in category_dir.iterdir() if inst.is_dir()]
                    print(f"   {category_dir.name}/: {', '.join(instances)}")
        else:
            print(f"   ❌ No instances directory found at {self.instances_dir}")

        # List algorithms
        print(f"\n🤖 Algorithms:")
        for alg_name in self.available_algorithms.keys():
            print(f"   {alg_name}")

        # List recent results
        print(f"\n📊 Recent Results:")
        if self.results_dir.exists():
            result_files = list(self.results_dir.glob("*.json"))
            result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            for result_file in result_files[:10]:  # Show 10 most recent
                mod_time = datetime.fromtimestamp(result_file.stat().st_mtime)
                print(f"   {result_file.name} ({mod_time.strftime('%Y-%m-%d %H:%M')})")

            if len(result_files) > 10:
                print(f"   ... and {len(result_files) - 10} more files")
        else:
            print(f"   ❌ No results directory found")

    def interactive_mode(self) -> None:
        """Run interactive command-line interface."""
        print(f"\n🎮 Interactive Mode")
        print("=" * 50)

        while True:
            print(f"\nWhat would you like to do?")
            print(f"1. 🔬 Run benchmark experiments")
            print(f"2. 🧪 Run single experiment")
            print(f"3. 📊 Analyze results")
            print(f"4. 📚 List available resources")
            print(f"5. ❓ Show system info")
            print(f"6. 🚪 Exit")

            choice = input(f"\nEnter choice (1-6): ").strip()

            if choice == "1":
                self._interactive_benchmark()
            elif choice == "2":
                self._interactive_single_experiment()
            elif choice == "3":
                self._interactive_analysis()
            elif choice == "4":
                self.list_available_resources()
            elif choice == "5":
                self._show_system_info()
            elif choice == "6":
                print(f"👋 Goodbye!")
                break
            else:
                print(f"❌ Invalid choice: {choice}")

    def _interactive_benchmark(self) -> None:
        """Interactive benchmark configuration."""
        print(f"\n🔬 Benchmark Configuration")
        print("-" * 30)

        profiles = ["quick", "comparison", "variants", "full"]
        print(f"Available profiles:")
        for i, profile in enumerate(profiles, 1):
            print(f"  {i}. {profile}")

        try:
            choice = int(input(f"Select profile (1-{len(profiles)}): "))
            if 1 <= choice <= len(profiles):
                profile = profiles[choice - 1]

                # Ask for customizations
                custom_instances = input(f"Custom instances pattern (Enter for default): ").strip()
                custom_algorithms = input(f"Custom algorithms (Enter for default): ").strip()
                if not custom_algorithms:
                    custom_algorithms = "greedy,genetic"

                verbose = input(f"Verbose output? (y/N): ").lower().startswith('y')

                self.run_benchmark(
                    profile=profile,
                    instances=custom_instances or None,
                    algorithms=custom_algorithms or None,
                    verbose=verbose
                )
            else:
                print(f"❌ Invalid selection")
        except ValueError:
            print(f"❌ Please enter a number")

    def _interactive_single_experiment(self) -> None:
        """Interactive single experiment configuration."""
        print(f"\n🧪 Single Experiment Configuration")
        print("-" * 35)

        algorithm = input(f"Algorithm name: ").strip()
        instance = input(f"Instance name/pattern: ").strip()
        variant = input(f"Algorithm variant (optional): ").strip()
        verbose = input(f"Verbose output? (Y/n): ").lower() not in ['n', 'no']

        if algorithm and instance:
            self.run_single_experiment(
                algorithm=algorithm,
                instance=instance,
                variant=variant or None,
                verbose=verbose
            )
        else:
            print(f"❌ Algorithm and instance are required")

    def _interactive_analysis(self) -> None:
        """Interactive analysis configuration."""
        print(f"\n📊 Analysis Configuration")
        print("-" * 25)

        focuses = ["overview", "comparison", "statistical"]
        print(f"Analysis focus:")
        for i, focus in enumerate(focuses, 1):
            print(f"  {i}. {focus}")

        try:
            choice = int(input(f"Select focus (1-{len(focuses)}): "))
            if 1 <= choice <= len(focuses):
                focus = focuses[choice - 1]
                export = input(f"Export CSV files? (Y/n): ").lower() not in ['n', 'no']

                self.run_analysis(focus=focus, export_csv=export)
            else:
                print(f"❌ Invalid selection")
        except ValueError:
            print(f"❌ Please enter a number")

    def _show_system_info(self) -> None:
        """Show system information and status."""
        print(f"\n🔧 System Information")
        print("=" * 30)
        print(f"Project root: {self.project_root}")
        print(f"Python path: {sys.executable}")
        print(f"Imports successful: {IMPORTS_SUCCESS}")
        print(f"Available algorithms: {len(self.available_algorithms)}")

        # Check directories
        dirs_to_check = [
            ("Instances", self.instances_dir),
            ("Results", self.results_dir),
        ]

        print(f"\nDirectory status:")
        for name, path in dirs_to_check:
            status = "✅ Exists" if path.exists() else "❌ Missing"
            print(f"  {name}: {status} ({path})")

        # Count files
        if self.instances_dir.exists():
            instance_count = len(
                [d for d in self.instances_dir.rglob("*") if d.is_dir() and d.parent != self.instances_dir])
            print(f"  Instance count: {instance_count}")

        if self.results_dir.exists():
            result_count = len(list(self.results_dir.glob("*.json")))
            print(f"  Result files: {result_count}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Course Scheduling Optimization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Interactive mode
  python main.py --mode benchmark --profile quick   # Quick benchmark
  python main.py --mode single --algorithm greedy --instance small_01
  python main.py --mode analyze --focus comparison  # Analyze results
  python main.py --list                            # List resources
        """
    )

    parser.add_argument("--mode", choices=["interactive", "benchmark", "single", "analyze"],
                        default="interactive", help="Operation mode")

    # Benchmark options
    parser.add_argument("--profile", choices=["quick", "comparison", "variants", "full"],
                        default="quick", help="Benchmark profile")
    parser.add_argument("--instances", type=str, help="Instance pattern override")
    parser.add_argument("--algorithms", type=str, help="Algorithm pattern override")
    parser.add_argument("--runs", type=int, help="Number of runs override")

    # Single experiment options
    parser.add_argument("--algorithm", type=str, help="Algorithm name for single experiment")
    parser.add_argument("--instance", type=str, help="Instance name for single experiment")
    parser.add_argument("--variant", type=str, help="Algorithm variant")

    # Analysis options
    parser.add_argument("--focus", choices=["overview", "comparison", "statistical"],
                        default="overview", help="Analysis focus")
    parser.add_argument("--no-export", action="store_true", help="Don't export CSV files")

    # General options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--list", action="store_true", help="List available resources and exit")

    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Initialize system
    system = CourseSchedulingSystem()

    # Handle list request
    if args.list:
        system.list_available_resources()
        return

    # Route to appropriate mode
    try:
        if args.mode == "interactive":
            system.interactive_mode()

        elif args.mode == "benchmark":
            success = system.run_benchmark(
                profile=args.profile,
                instances=args.instances,
                algorithms=args.algorithms,
                runs=args.runs,
                verbose=args.verbose
            )
            sys.exit(0 if success else 1)

        elif args.mode == "single":
            if not args.algorithm or not args.instance:
                print("❌ Single mode requires --algorithm and --instance")
                sys.exit(1)

            success = system.run_single_experiment(
                algorithm=args.algorithm,
                instance=args.instance,
                variant=args.variant,
                verbose=args.verbose
            )
            sys.exit(0 if success else 1)

        elif args.mode == "analyze":
            success = system.analyze_results(
                focus=args.focus,
                export_csv=not args.no_export
            )
            sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print(f"\n\n⏹️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()