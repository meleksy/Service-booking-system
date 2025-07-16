import os
import sys
import glob
import time
import json
from typing import List, Dict, Union, Optional, Type
from datetime import datetime
from pathlib import Path
import itertools

# Add src directories to path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.extend([
    os.path.join(project_root, 'src', 'utils'),
    os.path.join(project_root, 'src', 'models'),
    os.path.join(project_root, 'src', 'algorithms')
])

# Import required modules
from utils.data_loader import load_instance
from models.problem import CourseSchedulingProblem
from algorithms.base import Algorithm

# Import all algorithms
from algorithms.greedy import GreedyAlgorithm, create_greedy_variants
from algorithms.genetic import GeneticAlgorithm, create_genetic_variants
from algorithms.simulated_annealing import SimulatedAnnealingAlgorithm, create_sa_variants
from algorithms.tabu_search import TabuSearchAlgorithm, create_tabu_variants
from algorithms.vns import VariableNeighborhoodSearch, create_vns_variants


class ProgressTracker:
    """Simple progress tracker for batch experiments."""

    def __init__(self, total_experiments: int):
        self.total_experiments = total_experiments
        self.completed_experiments = 0
        self.start_time = time.time()
        self.failed_experiments = 0

    def update(self, success: bool = True):
        """Update progress after completing an experiment."""
        self.completed_experiments += 1
        if not success:
            self.failed_experiments += 1

        # Calculate progress
        progress_pct = (self.completed_experiments / self.total_experiments) * 100

        # Estimate remaining time
        elapsed_time = time.time() - self.start_time
        if self.completed_experiments > 0:
            time_per_experiment = elapsed_time / self.completed_experiments
            remaining_experiments = self.total_experiments - self.completed_experiments
            eta_seconds = remaining_experiments * time_per_experiment
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "Unknown"

        print(f"Progress: {self.completed_experiments}/{self.total_experiments} "
              f"({progress_pct:.1f}%) | Failed: {self.failed_experiments} | ETA: {eta_str}")

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"


class BatchRunner:
    """
    Batch runner for course scheduling experiments.

    Handles running multiple algorithms on multiple instances with
    automatic result collection and organization.
    """

    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize batch runner.

        Args:
            project_root: Root directory of the project
        """
        if project_root is None:
            self.project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        else:
            self.project_root = project_root

        self.instances_dir = os.path.join(self.project_root, 'data', 'instances')
        self.results_dir = os.path.join(self.project_root, 'data', 'results')

        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Algorithm registry
        self.algorithms = {
            'greedy': (GreedyAlgorithm, create_greedy_variants),
            'genetic': (GeneticAlgorithm, create_genetic_variants),
            'sa': (SimulatedAnnealingAlgorithm, create_sa_variants),
            'tabu': (TabuSearchAlgorithm, create_tabu_variants),
            'vns': (VariableNeighborhoodSearch, create_vns_variants)
        }

    def find_instances(self, pattern: Union[str, List[str]]) -> List[str]:
        """
        Find instance directories matching the given pattern(s).

        Args:
            pattern: Instance pattern(s) to match
                    - "small/*" - all instances in small category
                    - "*/small_01" - small_01 from all categories
                    - ["small_01", "medium_02"] - specific instances

        Returns:
            List of instance directory paths
        """
        if isinstance(pattern, str):
            patterns = [pattern]
        else:
            patterns = pattern

        instance_paths = []

        for pat in patterns:
            if '*' in pat:
                # Glob pattern
                search_path = os.path.join(self.instances_dir, pat)
                found_paths = glob.glob(search_path)
                # Filter to only directories
                found_paths = [p for p in found_paths if os.path.isdir(p)]
                instance_paths.extend(found_paths)
            else:
                # Direct path - try to find it
                possible_paths = []

                # Try as direct path
                direct_path = os.path.join(self.instances_dir, pat)
                if os.path.isdir(direct_path):
                    possible_paths.append(direct_path)

                # Try searching in subdirectories
                for category in os.listdir(self.instances_dir):
                    category_path = os.path.join(self.instances_dir, category)
                    if os.path.isdir(category_path):
                        instance_path = os.path.join(category_path, pat)
                        if os.path.isdir(instance_path):
                            possible_paths.append(instance_path)

                instance_paths.extend(possible_paths)

        # Remove duplicates and sort
        instance_paths = sorted(list(set(instance_paths)))

        if not instance_paths:
            print(f"Warning: No instances found matching pattern(s): {patterns}")

        return instance_paths

    def get_algorithm_configs(self, algorithm_specs: Union[str, List[str]]) -> List[Dict]:
        """
        Get algorithm configurations based on specifications.

        Args:
            algorithm_specs: Algorithm specification(s)
                          - "greedy" - greedy with default params
                          - "greedy_variants" - all greedy variants
                          - "all" - all algorithms with all variants
                          - ["greedy", "genetic_variants"] - mixed

        Returns:
            List of algorithm configurations
        """
        if isinstance(algorithm_specs, str):
            specs = [algorithm_specs]
        else:
            specs = algorithm_specs

        configs = []

        for spec in specs:
            if spec == "all":
                # All algorithms with all variants
                for alg_name, (alg_class, variant_func) in self.algorithms.items():
                    variants = variant_func()
                    for variant in variants:
                        configs.append({
                            'class': alg_class,
                            'name': variant.get('name', alg_name),
                            'params': variant
                        })

            elif spec.endswith("_variants"):
                # All variants of specific algorithm
                alg_name = spec.replace("_variants", "")
                if alg_name in self.algorithms:
                    alg_class, variant_func = self.algorithms[alg_name]
                    variants = variant_func()
                    for variant in variants:
                        configs.append({
                            'class': alg_class,
                            'name': variant.get('name', alg_name),
                            'params': variant
                        })
                else:
                    print(f"Warning: Unknown algorithm '{alg_name}'")

            elif spec in self.algorithms:
                # Single algorithm with default params
                alg_class, _ = self.algorithms[spec]
                configs.append({
                    'class': alg_class,
                    'name': spec,
                    'params': {}
                })

            else:
                print(f"Warning: Unknown algorithm specification '{spec}'")

        return configs

    def run_single_experiment(self, instance_path: str, algorithm_config: Dict,
                              run_id: int = 1, verbose: bool = False) -> Optional[Dict]:
        """
        Run a single experiment (one algorithm on one instance).

        Args:
            instance_path: Path to instance directory
            algorithm_config: Algorithm configuration dict
            run_id: Run identifier for multiple runs
            verbose: Whether to print detailed output

        Returns:
            Experiment result dict or None if failed
        """
        try:
            # Load instance
            loader = load_instance(instance_path)
            problem = CourseSchedulingProblem(loader)

            # Create algorithm
            algorithm_class = algorithm_config['class']
            algorithm_name = algorithm_config['name']
            params = algorithm_config['params'].copy()

            # Add experiment metadata to parameters
            params.update({
                'verbose': verbose,
                'instance_name': problem.instance_name,
                'run_id': f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id:03d}",
                'save_results': True,
                'results_dir': self.results_dir
            })

            # Run algorithm
            start_time = time.time()
            algorithm = algorithm_class(problem, **params)
            solution = algorithm.run()
            execution_time = time.time() - start_time

            # Collect results
            result = {
                'instance_name': problem.instance_name,
                'instance_path': instance_path,
                'algorithm_name': algorithm_name,
                'algorithm_class': algorithm_class.__name__,
                'run_id': run_id,
                'execution_time': execution_time,
                'success': True,
                'error_message': None
            }

            # Add solution metrics if available
            if algorithm.best_solution:
                x, y, u = algorithm.best_solution.to_arrays()
                is_feasible, violations = problem.validate_solution(x, y, u)

                detailed_breakdown = problem.calculate_detailed_objective(x, y, u)

                result.update({
                    'objective_value': algorithm.stats.best_objective,
                    'is_feasible': is_feasible,
                    'num_violations': len(violations) if violations else 0,
                    'scheduled_courses': len(algorithm.best_solution.get_all_assignments()),
                    'total_courses': problem.N,
                    'utilization': algorithm.best_solution.calculate_utilization(),
                    'detailed_breakdown': detailed_breakdown
                })

            return result

        except Exception as e:
            error_msg = str(e)
            if verbose:
                print(f"Error in experiment: {error_msg}")

            return {
                'instance_name': os.path.basename(instance_path),
                'instance_path': instance_path,
                'algorithm_name': algorithm_config.get('name', 'unknown'),
                'algorithm_class': algorithm_config['class'].__name__ if 'class' in algorithm_config else 'unknown',
                'run_id': run_id,
                'execution_time': 0.0,
                'success': False,
                'error_message': error_msg,
                'objective_value': None,
                'is_feasible': False,
                'num_violations': float('inf')
            }

    def run_batch(self, instances: Union[str, List[str]],
                  algorithms: Union[str, List[str]],
                  runs_per_combination: int = 1,
                  verbose: bool = False,
                  save_summary: bool = True) -> List[Dict]:
        """
        Run batch experiments.

        Args:
            instances: Instance pattern(s) to run
            algorithms: Algorithm specification(s) to run
            runs_per_combination: Number of runs per (instance, algorithm) combination
            verbose: Whether to print detailed output for each run
            save_summary: Whether to save batch summary

        Returns:
            List of experiment results
        """
        print(f"=== STARTING BATCH EXPERIMENTS ===")
        print(f"Timestamp: {datetime.now().isoformat()}")

        # Find instances and algorithms
        instance_paths = self.find_instances(instances)
        algorithm_configs = self.get_algorithm_configs(algorithms)

        if not instance_paths:
            print(f"ERROR: No instances found for pattern '{instances}'")
            print(f"Searched in: {self.instances_dir}")
            print(f"Available directories:")
            if os.path.exists(self.instances_dir):
                for category in os.listdir(self.instances_dir):
                    category_path = os.path.join(self.instances_dir, category)
                    if os.path.isdir(category_path):
                        print(f"  {category}/")
                        for instance in os.listdir(category_path):
                            instance_path = os.path.join(category_path, instance)
                            if os.path.isdir(instance_path):
                                print(f"    {instance}/")
            return []

        if not algorithm_configs:
            print(f"ERROR: No algorithms found for pattern '{algorithms}'")
            return []

        print(f"Found {len(instance_paths)} instances:")
        for path in instance_paths:
            print(f"  - {os.path.basename(path)}")

        print(f"Found {len(algorithm_configs)} algorithm configurations:")
        for config in algorithm_configs:
            print(f"  - {config['name']}")

        # Calculate total experiments
        total_experiments = len(instance_paths) * len(algorithm_configs) * runs_per_combination
        print(f"Total experiments to run: {total_experiments}")
        print(f"Runs per combination: {runs_per_combination}")
        print()

        # Initialize progress tracker
        progress = ProgressTracker(total_experiments)

        # Run all experiments
        results = []
        experiment_count = 0

        for instance_path in instance_paths:
            for algorithm_config in algorithm_configs:
                for run_id in range(1, runs_per_combination + 1):
                    experiment_count += 1

                    if not verbose:
                        print(f"Running experiment {experiment_count}/{total_experiments}: "
                              f"{os.path.basename(instance_path)} + {algorithm_config['name']} "
                              f"(run {run_id}/{runs_per_combination})")

                    # Run single experiment
                    result = self.run_single_experiment(
                        instance_path, algorithm_config, run_id, verbose
                    )

                    if result:
                        results.append(result)
                        progress.update(success=result['success'])
                    else:
                        progress.update(success=False)

                    if not verbose:
                        if result and result['success']:
                            obj_val = result.get('objective_value', 'N/A')
                            feasible = result.get('is_feasible', False)
                            print(f"  Result: obj={obj_val}, feasible={feasible}")
                        else:
                            print(f"  Result: FAILED")
                        print()

        # Save batch summary
        if save_summary:
            self._save_batch_summary(results, instances, algorithms, runs_per_combination)

        print(f"=== BATCH EXPERIMENTS COMPLETED ===")
        print(f"Total experiments: {total_experiments}")
        print(f"Successful: {total_experiments - progress.failed_experiments}")
        print(f"Failed: {progress.failed_experiments}")
        print(f"Total time: {progress._format_time(time.time() - progress.start_time)}")
        print(f"Results directory: {self.results_dir}")

        # List generated result files
        if save_summary:
            result_files = glob.glob(os.path.join(self.results_dir, "*.json"))
            recent_files = [f for f in result_files if
                            os.path.getmtime(f) > progress.start_time - 60]  # Files from last run

            print(f"Generated {len(recent_files)} result files:")
            for file_path in sorted(recent_files)[-10:]:  # Show last 10
                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                print(f"  - {filename} ({file_size} bytes)")

        return results

    def _save_batch_summary(self, results: List[Dict], instances: Union[str, List[str]],
                            algorithms: Union[str, List[str]], runs_per_combination: int) -> None:
        """Save batch experiment summary."""
        summary = {
            'batch_info': {
                'timestamp': datetime.now().isoformat(),
                'instances_pattern': instances,
                'algorithms_pattern': algorithms,
                'runs_per_combination': runs_per_combination,
                'total_experiments': len(results)
            },
            'results_summary': {
                'successful_runs': sum(1 for r in results if r['success']),
                'failed_runs': sum(1 for r in results if not r['success']),
                'unique_instances': len(set(r['instance_name'] for r in results)),
                'unique_algorithms': len(set(r['algorithm_name'] for r in results))
            },
            'detailed_results': results
        }

        # Add aggregated statistics
        successful_results = [r for r in results if r['success'] and r.get('objective_value') is not None]
        if successful_results:
            obj_values = [r['objective_value'] for r in successful_results]

            # Find best overall and best feasible
            best_overall = max(successful_results, key=lambda x: x['objective_value'])
            feasible_results = [r for r in successful_results if r.get('is_feasible', False)]

            summary['statistics'] = {
                'best_objective': max(obj_values),
                'worst_objective': min(obj_values),
                'average_objective': sum(obj_values) / len(obj_values),
                'feasible_solutions': sum(1 for r in successful_results if r.get('is_feasible', False)),
                'average_execution_time': sum(r['execution_time'] for r in successful_results) / len(
                    successful_results),

                # Best overall solution
                'best_solution': {
                    'algorithm': best_overall['algorithm_name'],
                    'instance': best_overall['instance_name'],
                    'objective_value': best_overall['objective_value'],
                    'is_feasible': best_overall.get('is_feasible', False),
                    'execution_time': best_overall['execution_time'],
                    'detailed_breakdown': best_overall.get('detailed_breakdown', {}),
                    'scheduled_courses': best_overall.get('scheduled_courses', 0),
                    'total_courses': best_overall.get('total_courses', 0)
                }
            }

            # Best feasible solution (if any)
            if feasible_results:
                best_feasible = max(feasible_results, key=lambda x: x['objective_value'])
                summary['statistics']['best_feasible_solution'] = {
                    'algorithm': best_feasible['algorithm_name'],
                    'instance': best_feasible['instance_name'],
                    'objective_value': best_feasible['objective_value'],
                    'is_feasible': True,
                    'execution_time': best_feasible['execution_time'],
                    'detailed_breakdown': best_feasible.get('detailed_breakdown', {}),
                    'scheduled_courses': best_feasible.get('scheduled_courses', 0),
                    'total_courses': best_feasible.get('total_courses', 0),
                    'gap_to_best_overall': best_overall['objective_value'] - best_feasible['objective_value']
                }
            else:
                summary['statistics']['best_feasible_solution'] = None
        else:
            # Add empty statistics for failed batches
            summary['statistics'] = {
                'best_objective': None,
                'worst_objective': None,
                'average_objective': None,
                'feasible_solutions': 0,
                'average_execution_time': 0.0,
                'note': 'No successful results with valid objectives'
            }

        # Save summary
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_filename = f"batch_summary_{timestamp}.json"
        summary_path = os.path.join(self.results_dir, summary_filename)

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"Batch summary saved to: {summary_path}")

        if successful_results:
            print(f"\n📊 BEST SOLUTIONS SUMMARY:")
            print(f"========================")

            print(f"🏆 Best Overall:")
            print(f"   Algorithm: {best_overall['algorithm_name']}")
            print(f"   Objective: {best_overall['objective_value']:.2f}")
            print(f"   Feasible: {'✅' if best_overall.get('is_feasible') else '❌'}")
            print(f"   Time: {best_overall['execution_time']:.3f}s")

            if feasible_results:
                best_feasible = max(feasible_results, key=lambda x: x['objective_value'])
                gap = best_overall['objective_value'] - best_feasible['objective_value']
                gap_percent = (gap / best_overall['objective_value'] * 100) if best_overall[
                                                                                   'objective_value'] > 0 else 0

                print(f"\n✅ Best Feasible:")
                print(f"   Algorithm: {best_feasible['algorithm_name']}")
                print(f"   Objective: {best_feasible['objective_value']:.2f}")
                print(f"   Gap to best: {gap:.2f} ({gap_percent:.1f}%)")
                print(f"   Time: {best_feasible['execution_time']:.3f}s")

                # Print detailed breakdown if available
                if best_feasible.get('detailed_breakdown'):
                    breakdown = best_feasible['detailed_breakdown']
                    print(f"   Breakdown:")
                    print(f"     Revenue: {breakdown.get('zysk', 0):.0f}")
                    print(f"     Room usage: {breakdown.get('koszt_uzytkowania', 0):.0f}")
                    print(f"     Room idle: {breakdown.get('koszt_niewykorzystania', 0):.0f}")
                    print(f"     Instructors: {breakdown.get('koszt_instruktorow', 0):.0f}")
            else:
                print(f"\n❌ No feasible solutions found!")

    def run_quick_test(self, instance_pattern: str = "small/small_01",
                       algorithm: str = "greedy") -> None:
        """Run quick test with single instance and algorithm."""
        print("=== QUICK TEST ===")
        results = self.run_batch(
            instances=instance_pattern,
            algorithms=algorithm,
            runs_per_combination=1,
            verbose=True
        )

        if results and results[0]['success']:
            print("✅ Quick test successful!")
        else:
            print("❌ Quick test failed!")

    def run_algorithm_comparison(self, instance_pattern: str = "small/*",
                                 algorithms: List[str] = None) -> None:
        """Run comparison of all algorithms on given instances."""
        if algorithms is None:
            algorithms = ["greedy", "genetic", "sa", "tabu", "vns"]

        print("=== ALGORITHM COMPARISON ===")
        results = self.run_batch(
            instances=instance_pattern,
            algorithms=algorithms,
            runs_per_combination=1,
            verbose=False
        )

        # Print quick comparison
        if results:
            print("\nQuick comparison (best objectives):")
            by_algorithm = {}
            for result in results:
                if result['success'] and result.get('objective_value'):
                    alg_name = result['algorithm_name']
                    if alg_name not in by_algorithm:
                        by_algorithm[alg_name] = []
                    by_algorithm[alg_name].append(result['objective_value'])

            for alg_name, objectives in by_algorithm.items():
                avg_obj = sum(objectives) / len(objectives)
                print(f"  {alg_name}: avg={avg_obj:.2f} (from {len(objectives)} runs)")

    def run_full_benchmark(self) -> None:
        """Run comprehensive benchmark with all algorithms and instances."""
        print("=== FULL BENCHMARK ===")
        print("This will run all algorithms on all instances with multiple runs.")
        print("This may take a long time!")

        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Benchmark cancelled.")
            return

        self.run_batch(
            instances="*/*",  # All instances
            algorithms="all",  # All algorithm variants
            runs_per_combination=3,
            verbose=False
        )


def main():
    """Main function with example usage."""
    runner = BatchRunner()

    print("Course Scheduling Batch Runner")
    print("==============================")
    print()
    print("Available commands:")
    print("1. Quick test")
    print("2. Algorithm comparison")
    print("3. Custom batch")
    print("4. Full benchmark")
    print("5. Exit")

    while True:
        choice = input("\nEnter choice (1-5): ").strip()

        if choice == "1":
            runner.run_quick_test()

        elif choice == "2":
            pattern = input("Instance pattern (default: small/*): ").strip()
            if not pattern:
                pattern = "small/*"
            runner.run_algorithm_comparison(pattern)

        elif choice == "3":
            instances = input("Instance pattern (e.g., small/*, */small_01): ").strip()
            algorithms = input("Algorithms (e.g., greedy, genetic_variants, all): ").strip()
            runs = input("Runs per combination (default: 1): ").strip()

            if not instances or not algorithms:
                print("Both instances and algorithms are required!")
                continue

            runs = int(runs) if runs.isdigit() else 1

            runner.run_batch(
                instances=instances,
                algorithms=algorithms.split(','),
                runs_per_combination=runs
            )

        elif choice == "4":
            runner.run_full_benchmark()

        elif choice == "5":
            print("Goodbye!")
            break

        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()