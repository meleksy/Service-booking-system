import os
import json
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics
from datetime import datetime
import scipy.stats as stats
import os
from pathlib import Path


@dataclass
class AlgorithmPerformance:
    """Performance metrics for an algorithm on specific instances."""
    algorithm_name: str
    instance_name: str
    runs: List[Dict]

    # Aggregated metrics
    avg_objective: float = 0.0
    std_objective: float = 0.0
    best_objective: float = 0.0
    worst_objective: float = 0.0
    median_objective: float = 0.0

    avg_time: float = 0.0
    std_time: float = 0.0

    success_rate: float = 0.0
    feasibility_rate: float = 0.0

    avg_utilization: float = 0.0
    avg_scheduled_courses: float = 0.0

    # Statistical confidence
    confidence_interval_95: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        """Calculate aggregated metrics from runs."""
        if not self.runs:
            return

        successful_runs = [r for r in self.runs if r.get('success', False)]
        feasible_runs = [r for r in successful_runs if r.get('is_feasible', False)]

        if successful_runs:
            objectives = [r.get('objective_value', 0) for r in successful_runs if r.get('objective_value') is not None]
            times = [r.get('execution_time', 0) for r in successful_runs]

            if objectives:
                self.avg_objective = statistics.mean(objectives)
                self.std_objective = statistics.stdev(objectives) if len(objectives) > 1 else 0.0
                self.best_objective = max(objectives)
                self.worst_objective = min(objectives)
                self.median_objective = statistics.median(objectives)

                # 95% confidence interval
                if len(objectives) > 1:
                    confidence_level = 0.95
                    degrees_freedom = len(objectives) - 1
                    sample_mean = statistics.mean(objectives)
                    sample_standard_error = stats.sem(objectives)
                    confidence_interval = stats.t.interval(confidence_level, degrees_freedom, sample_mean,
                                                           sample_standard_error)
                    self.confidence_interval_95 = confidence_interval

            if times:
                self.avg_time = statistics.mean(times)
                self.std_time = statistics.stdev(times) if len(times) > 1 else 0.0

            utilizations = [r.get('utilization', {}).get('room_utilization', 0) for r in successful_runs
                            if r.get('utilization')]
            if utilizations:
                self.avg_utilization = statistics.mean(utilizations)

            scheduled = [r.get('scheduled_courses', 0) for r in successful_runs]
            if scheduled:
                self.avg_scheduled_courses = statistics.mean(scheduled)

        self.success_rate = len(successful_runs) / len(self.runs) if self.runs else 0.0
        self.feasibility_rate = len(feasible_runs) / len(self.runs) if self.runs else 0.0


@dataclass
class ComparisonResult:
    """Result of statistical comparison between two algorithms."""
    algorithm_a: str
    algorithm_b: str
    instance_name: str

    # Statistical test results
    is_significant: bool = False
    p_value: float = 1.0
    test_statistic: float = 0.0
    test_name: str = ""

    # Effect size
    effect_size: float = 0.0
    effect_size_interpretation: str = ""

    # Practical difference
    mean_difference: float = 0.0
    relative_improvement: float = 0.0

    winner: str = "tie"


class MetricsCalculator:
    """Calculator for various performance metrics and comparisons."""

    @staticmethod
    def calculate_relative_gap(algorithm_result: float, best_known: float) -> float:
        """Calculate relative gap to best known solution."""
        if best_known == 0:
            return 0.0
        return (best_known - algorithm_result) / best_known * 100

    @staticmethod
    def calculate_efficiency_score(objective: float, time: float, baseline_time: float = 1.0) -> float:
        """Calculate efficiency score (objective per unit time)."""
        if time <= 0:
            return 0.0
        normalized_time = time / baseline_time
        return objective / normalized_time

    @staticmethod
    def calculate_robustness_score(results: List[float]) -> float:
        """Calculate robustness score (1 - coefficient_of_variation)."""
        if not results or len(results) < 2:
            return 1.0

        mean_val = statistics.mean(results)
        if mean_val == 0:
            return 0.0

        std_val = statistics.stdev(results)
        cv = std_val / mean_val
        return max(0.0, 1.0 - cv)

    @staticmethod
    def calculate_dominance_score(algorithm_results: Dict[str, List[float]],
                                  algorithm_name: str) -> float:
        """Calculate how often this algorithm dominates others."""
        if algorithm_name not in algorithm_results:
            return 0.0

        alg_results = algorithm_results[algorithm_name]
        other_algorithms = {k: v for k, v in algorithm_results.items() if k != algorithm_name}

        if not other_algorithms or not alg_results:
            return 0.0

        dominance_count = 0
        total_comparisons = 0

        for other_name, other_results in other_algorithms.items():
            min_length = min(len(alg_results), len(other_results))
            for i in range(min_length):
                if alg_results[i] > other_results[i]:
                    dominance_count += 1
                total_comparisons += 1

        return dominance_count / total_comparisons if total_comparisons > 0 else 0.0


class StatisticalAnalyzer:
    """Performs statistical analysis and hypothesis testing."""

    @staticmethod
    def wilcoxon_test(results_a: List[float], results_b: List[float]) -> ComparisonResult:
        """Perform Wilcoxon signed-rank test for paired samples."""
        if len(results_a) != len(results_b) or len(results_a) < 3:
            return ComparisonResult("", "", "", p_value=1.0, test_name="wilcoxon_insufficient_data")

        try:
            statistic, p_value = stats.wilcoxon(results_a, results_b, alternative='two-sided')

            # Effect size (r = Z / sqrt(N))
            z_score = stats.norm.ppf(1 - p_value / 2)  # Approximate Z from p-value
            n = len(results_a)
            effect_size = abs(z_score) / np.sqrt(n)

            # Interpret effect size
            if effect_size < 0.1:
                effect_interpretation = "negligible"
            elif effect_size < 0.3:
                effect_interpretation = "small"
            elif effect_size < 0.5:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"

            mean_diff = statistics.mean(results_a) - statistics.mean(results_b)

            return ComparisonResult(
                algorithm_a="A", algorithm_b="B", instance_name="",
                is_significant=p_value < 0.05,
                p_value=p_value,
                test_statistic=statistic,
                test_name="wilcoxon",
                effect_size=effect_size,
                effect_size_interpretation=effect_interpretation,
                mean_difference=mean_diff,
                winner="A" if mean_diff > 0 and p_value < 0.05 else "B" if mean_diff < 0 and p_value < 0.05 else "tie"
            )
        except Exception as e:
            return ComparisonResult("", "", "", p_value=1.0, test_name=f"wilcoxon_error_{str(e)}")

    @staticmethod
    def t_test(results_a: List[float], results_b: List[float]) -> ComparisonResult:
        """Perform independent t-test."""
        if len(results_a) < 2 or len(results_b) < 2:
            return ComparisonResult("", "", "", p_value=1.0, test_name="t_test_insufficient_data")

        try:
            statistic, p_value = stats.ttest_ind(results_a, results_b)

            # Cohen's d effect size
            pooled_std = np.sqrt(((len(results_a) - 1) * np.var(results_a, ddof=1) +
                                  (len(results_b) - 1) * np.var(results_b, ddof=1)) /
                                 (len(results_a) + len(results_b) - 2))

            if pooled_std > 0:
                cohens_d = (statistics.mean(results_a) - statistics.mean(results_b)) / pooled_std
            else:
                cohens_d = 0.0

            # Interpret effect size
            abs_d = abs(cohens_d)
            if abs_d < 0.2:
                effect_interpretation = "negligible"
            elif abs_d < 0.5:
                effect_interpretation = "small"
            elif abs_d < 0.8:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"

            mean_diff = statistics.mean(results_a) - statistics.mean(results_b)

            return ComparisonResult(
                algorithm_a="A", algorithm_b="B", instance_name="",
                is_significant=p_value < 0.05,
                p_value=p_value,
                test_statistic=statistic,
                test_name="t_test",
                effect_size=abs_d,
                effect_size_interpretation=effect_interpretation,
                mean_difference=mean_diff,
                winner="A" if mean_diff > 0 and p_value < 0.05 else "B" if mean_diff < 0 and p_value < 0.05 else "tie"
            )
        except Exception as e:
            return ComparisonResult("", "", "", p_value=1.0, test_name=f"t_test_error_{str(e)}")


class ResultsAnalyzer:
    """
    Main class for analyzing experimental results.

    Loads results from JSON files, calculates performance metrics,
    performs statistical analysis, and generates reports.
    """

    # Na początku klasy ResultsAnalyzer
    def __init__(self, results_dir: str = None):
        if results_dir is None:
            # Znajdź katalog główny projektu (tam gdzie jest src/)
            current_file = Path(__file__)  # metrics.py
            project_root = current_file.parent.parent.parent  # wyjdź z utils -> src -> root
            results_dir = project_root / "data" / "results"

        self.results_dir = Path(results_dir)
        """
        Initialize results analyzer.

        Args:
            results_dir: Directory containing result JSON files
        """
        self.results_dir = results_dir
        self.raw_results: List[Dict] = []
        self.algorithm_performances: Dict[Tuple[str, str], AlgorithmPerformance] = {}
        self.metrics_calculator = MetricsCalculator()
        self.statistical_analyzer = StatisticalAnalyzer()

    def load_results(self, pattern: str = "*.json", exclude_batch_summaries: bool = True) -> None:
        """
        Load results from JSON files.

        Args:
            pattern: File pattern to match
            exclude_batch_summaries: Whether to exclude batch summary files
        """
        search_pattern = os.path.join(self.results_dir, pattern)
        print(f"Searching for results in: {self.results_dir}")
        print(f"Search pattern: {search_pattern}")

        result_files = glob.glob(search_pattern)
        print(f"Found files before filtering: {len(result_files)}")

        if exclude_batch_summaries:
            result_files = [f for f in result_files if not os.path.basename(f).startswith('batch_summary')]
            print(f"Found files after filtering batch summaries: {len(result_files)}")

        # Show available files for debugging
        if not result_files:
            print("No result files found. Checking directory contents:")
            if os.path.exists(self.results_dir):
                all_files = os.listdir(self.results_dir)
                print(f"Files in {self.results_dir}:")
                for f in all_files:
                    file_path = os.path.join(self.results_dir, f)
                    if os.path.isfile(file_path):
                        print(f"  - {f}")
                if not all_files:
                    print("  (directory is empty)")
            else:
                print(f"  Directory {self.results_dir} does not exist!")
                return

        print(f"Loading results from {len(result_files)} files...")

        self.raw_results = []
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                    result['file_path'] = file_path
                    self.raw_results.append(result)
                    print(f"  ✓ Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  ✗ Failed to load {file_path}: {e}")

        print(f"Successfully loaded {len(self.raw_results)} results")
        if self.raw_results:
            self._process_results()
        else:
            print("No results to process!")

    def _process_results(self) -> None:
        """Process raw results into structured performance data."""
        # Group results by (algorithm, instance)
        grouped_results = defaultdict(list)

        for result in self.raw_results:
            metadata = result.get('metadata', {})
            algorithm_name = metadata.get('algorithm', 'unknown')
            instance_name = metadata.get('instance_name', 'unknown')

            # Extract relevant metrics
            stats = result.get('statistics', {})
            solution = result.get('solution', {})

            processed_result = {
                'success': solution is not None,
                'objective_value': stats.get('best_objective'),
                'execution_time': stats.get('execution_time', 0),
                'is_feasible': solution.get('is_feasible', False) if solution else False,
                'num_violations': solution.get('num_violations', float('inf')) if solution else float('inf'),
                'scheduled_courses': solution.get('scheduled_courses', 0) if solution else 0,
                'total_courses': result.get('problem_info', {}).get('num_courses', 0),
                'utilization': solution.get('utilization', {}) if solution else {},

                'detailed_breakdown': solution.get('detailed_breakdown', {}) if solution else {}
            }

            grouped_results[(algorithm_name, instance_name)].append(processed_result)

        # Create AlgorithmPerformance objects
        self.algorithm_performances = {}
        for (algorithm_name, instance_name), runs in grouped_results.items():
            performance = AlgorithmPerformance(
                algorithm_name=algorithm_name,
                instance_name=instance_name,
                runs=runs
            )
            self.algorithm_performances[(algorithm_name, instance_name)] = performance

    def get_algorithm_summary(self, algorithm_name: str) -> Dict:
        """Get summary statistics for an algorithm across all instances."""
        algorithm_performances = [
            perf for (alg, inst), perf in self.algorithm_performances.items()
            if alg == algorithm_name
        ]

        if not algorithm_performances:
            return {}

        objectives = [perf.avg_objective for perf in algorithm_performances if perf.avg_objective > 0]
        times = [perf.avg_time for perf in algorithm_performances if perf.avg_time > 0]
        success_rates = [perf.success_rate for perf in algorithm_performances]
        feasibility_rates = [perf.feasibility_rate for perf in algorithm_performances]

        summary = {
            'algorithm_name': algorithm_name,
            'instances_tested': len(algorithm_performances),
            'total_runs': sum(len(perf.runs) for perf in algorithm_performances),

            'avg_objective': statistics.mean(objectives) if objectives else 0,
            'std_objective': statistics.stdev(objectives) if len(objectives) > 1 else 0,
            'best_objective': max(objectives) if objectives else 0,

            'avg_time': statistics.mean(times) if times else 0,
            'std_time': statistics.stdev(times) if len(times) > 1 else 0,

            'avg_success_rate': statistics.mean(success_rates) if success_rates else 0,
            'avg_feasibility_rate': statistics.mean(feasibility_rates) if feasibility_rates else 0,

            'robustness_score': self.metrics_calculator.calculate_robustness_score(objectives),
        }

        return summary

    def compare_algorithms(self, algorithm_a: str, algorithm_b: str,
                           instance_name: Optional[str] = None,
                           metric: str = 'objective_value') -> List[ComparisonResult]:
        """
        Compare two algorithms statistically.

        Args:
            algorithm_a, algorithm_b: Algorithm names to compare
            instance_name: Specific instance to compare (None for all)
            metric: Metric to compare ('objective_value', 'execution_time')

        Returns:
            List of comparison results (one per instance)
        """
        comparisons = []

        # Filter performances
        if instance_name:
            instances_to_compare = [instance_name]
        else:
            all_instances = set()
            for (alg, inst), _ in self.algorithm_performances.items():
                if alg in [algorithm_a, algorithm_b]:
                    all_instances.add(inst)
            instances_to_compare = list(all_instances)

        for instance in instances_to_compare:
            perf_a = self.algorithm_performances.get((algorithm_a, instance))
            perf_b = self.algorithm_performances.get((algorithm_b, instance))

            if not perf_a or not perf_b:
                continue

            # Extract metric values
            if metric == 'objective_value':
                values_a = [r.get('objective_value', 0) for r in perf_a.runs
                            if r.get('success') and r.get('objective_value') is not None]
                values_b = [r.get('objective_value', 0) for r in perf_b.runs
                            if r.get('success') and r.get('objective_value') is not None]
            elif metric == 'execution_time':
                values_a = [r.get('execution_time', 0) for r in perf_a.runs if r.get('success')]
                values_b = [r.get('execution_time', 0) for r in perf_b.runs if r.get('success')]
            else:
                continue

            if not values_a or not values_b:
                continue

            # Perform statistical test
            if len(values_a) == len(values_b) and len(values_a) >= 3:
                comparison = self.statistical_analyzer.wilcoxon_test(values_a, values_b)
            else:
                comparison = self.statistical_analyzer.t_test(values_a, values_b)

            # Update comparison with correct names
            comparison.algorithm_a = algorithm_a
            comparison.algorithm_b = algorithm_b
            comparison.instance_name = instance

            # Calculate relative improvement
            if metric == 'objective_value':
                mean_a = statistics.mean(values_a)
                mean_b = statistics.mean(values_b)
                if mean_b > 0:
                    comparison.relative_improvement = (mean_a - mean_b) / mean_b * 100

            comparisons.append(comparison)

        return comparisons

    def generate_performance_ranking(self, metric: str = 'avg_objective') -> List[Dict]:
        """Generate ranking of algorithms by specified metric."""
        algorithm_summaries = {}

        # Get summaries for all algorithms
        algorithms = set(alg for (alg, inst), _ in self.algorithm_performances.items())
        for algorithm in algorithms:
            summary = self.get_algorithm_summary(algorithm)
            if summary:
                algorithm_summaries[algorithm] = summary

        # Sort by metric
        reverse_sort = metric in ['avg_objective', 'best_objective', 'avg_success_rate',
                                  'avg_feasibility_rate', 'robustness_score']

        ranking = sorted(algorithm_summaries.values(),
                         key=lambda x: x.get(metric, 0),
                         reverse=reverse_sort)

        # Add rank
        for i, alg_summary in enumerate(ranking):
            alg_summary['rank'] = i + 1

        return ranking

    def print_cost_breakdown_summary(self) -> None:
        """Print summary of cost breakdowns across algorithms."""
        print(f"\n📊 DETAILED COST BREAKDOWN ANALYSIS")
        print("=" * 60)

        # Group by algorithm
        algorithm_breakdowns = defaultdict(list)

        for (algorithm, instance), perf in self.algorithm_performances.items():
            for run in perf.runs:
                if run.get('success') and run.get('detailed_breakdown'):
                    breakdown = run['detailed_breakdown']
                    algorithm_breakdowns[algorithm].append(breakdown)

        # Calculate averages for each algorithm
        for algorithm, breakdowns in algorithm_breakdowns.items():
            if not breakdowns:
                continue

            avg_breakdown = {}
            for key in ['zysk', 'koszt_uzytkowania', 'koszt_niewykorzystania', 'koszt_instruktorow', 'funkcja']:
                values = [b.get(key, 0) for b in breakdowns if key in b]
                avg_breakdown[key] = sum(values) / len(values) if values else 0

            print(f"\n{algorithm}:")
            print(f"  Zysk (revenue):           {avg_breakdown['zysk']:>8.0f}")
            print(f"  Koszt użytkowania sal:    {avg_breakdown['koszt_uzytkowania']:>8.0f}")
            print(f"  Koszt niewykorzystania:   {avg_breakdown['koszt_niewykorzystania']:>8.0f}")
            print(f"  Koszt instruktorów:       {avg_breakdown['koszt_instruktorow']:>8.0f}")
            print(f"  Funkcja celu (TOTAL):     {avg_breakdown['funkcja']:>8.0f}")

            # Calculate cost ratios
            total_costs = (avg_breakdown['koszt_uzytkowania'] +
                           avg_breakdown['koszt_niewykorzystania'] +
                           avg_breakdown['koszt_instruktorow'])
            if total_costs > 0:
                cost_efficiency = avg_breakdown['zysk'] / total_costs
                print(f"  Efektywność (zysk/koszt): {cost_efficiency:>8.2f}")

    def export_to_csv(self, filename: str, include_raw_data: bool = False) -> None:
        """Export results to CSV file."""
        if include_raw_data:
            # Export all raw results
            rows = []
            for result in self.raw_results:
                metadata = result.get('metadata', {})
                stats = result.get('statistics', {})
                solution = result.get('solution', {})

                row = {
                    'algorithm': metadata.get('algorithm', 'unknown'),
                    'instance': metadata.get('instance_name', 'unknown'),
                    'timestamp': metadata.get('timestamp', ''),
                    'objective_value': stats.get('best_objective'),
                    'execution_time': stats.get('execution_time', 0),
                    'iterations': stats.get('iterations', 0),
                    'evaluations': stats.get('evaluations', 0),
                    'is_feasible': solution.get('is_feasible', False) if solution else False,
                    'scheduled_courses': solution.get('scheduled_courses', 0) if solution else 0,
                    'total_courses': result.get('problem_info', {}).get('num_courses', 0)
                }
                rows.append(row)

            df = pd.DataFrame(rows)
        else:
            # Export aggregated performance data
            rows = []
            for (algorithm, instance), perf in self.algorithm_performances.items():
                row = {
                    'algorithm': algorithm,
                    'instance': instance,
                    'avg_objective': perf.avg_objective,
                    'std_objective': perf.std_objective,
                    'best_objective': perf.best_objective,
                    'worst_objective': perf.worst_objective,
                    'avg_time': perf.avg_time,
                    'success_rate': perf.success_rate,
                    'feasibility_rate': perf.feasibility_rate,
                    'runs_count': len(perf.runs)
                }
                rows.append(row)

            df = pd.DataFrame(rows)

        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")

    def analyze_feasibility_gap(self) -> None:
        """Analyze gap between best overall and best feasible solutions."""
        print(f"\n🔍 FEASIBILITY GAP ANALYSIS")
        print("=" * 50)

        # Group by instance
        instance_analysis = defaultdict(lambda: {'best_overall': None, 'best_feasible': None})

        for (algorithm, instance), perf in self.algorithm_performances.items():
            for run in perf.runs:
                if not run.get('success') or run.get('objective_value') is None:
                    continue

                objective = run['objective_value']
                is_feasible = run.get('is_feasible', False)

                # Track best overall for this instance
                if (instance_analysis[instance]['best_overall'] is None or
                        objective > instance_analysis[instance]['best_overall']['objective']):
                    instance_analysis[instance]['best_overall'] = {
                        'algorithm': algorithm,
                        'objective': objective,
                        'feasible': is_feasible
                    }

                # Track best feasible for this instance
                if (is_feasible and
                        (instance_analysis[instance]['best_feasible'] is None or
                         objective > instance_analysis[instance]['best_feasible']['objective'])):
                    instance_analysis[instance]['best_feasible'] = {
                        'algorithm': algorithm,
                        'objective': objective,
                        'feasible': True
                    }

        # Print analysis
        for instance, data in instance_analysis.items():
            print(f"\nInstance: {instance}")

            if data['best_overall']:
                best_overall = data['best_overall']
                print(f"  Best Overall: {best_overall['algorithm']} = {best_overall['objective']:.2f} "
                      f"({'✅' if best_overall['feasible'] else '❌'})")

                if data['best_feasible']:
                    best_feasible = data['best_feasible']
                    gap = best_overall['objective'] - best_feasible['objective']
                    gap_percent = (gap / best_overall['objective'] * 100) if best_overall['objective'] > 0 else 0

                    print(f"  Best Feasible: {best_feasible['algorithm']} = {best_feasible['objective']:.2f}")
                    print(f"  Feasibility Gap: {gap:.2f} ({gap_percent:.1f}%)")

                    if gap_percent > 10:
                        print(f"  ⚠️  Large feasibility gap - consider algorithm tuning")
                    elif gap_percent < 1:
                        print(f"  ✅ Excellent feasible solution quality")
                else:
                    print(f"  ❌ No feasible solutions found!")
            else:
                print(f"  No valid results")

    def print_summary_report(self) -> None:
        """Print comprehensive summary report."""
        print("=" * 80)
        print("EXPERIMENTAL RESULTS SUMMARY")
        print("=" * 80)

        # Overall statistics
        total_results = len(self.raw_results)
        algorithms = set(alg for (alg, inst), _ in self.algorithm_performances.items())
        instances = set(inst for (alg, inst), _ in self.algorithm_performances.items())

        print(f"Total results analyzed: {total_results}")
        print(f"Algorithms tested: {len(algorithms)}")
        print(f"Instances tested: {len(instances)}")
        print(f"Algorithm-instance combinations: {len(self.algorithm_performances)}")
        print()

        # Algorithm ranking
        print("ALGORITHM RANKING (by average objective value):")
        print("-" * 80)
        ranking = self.generate_performance_ranking('avg_objective')

        print(f"{'Rank':<4} {'Algorithm':<25} {'Avg Obj':<12} {'Success Rate':<12} {'Avg Time':<10}")
        print("-" * 80)
        for alg in ranking:
            print(f"{alg['rank']:<4} {alg['algorithm_name']:<25} "
                  f"{alg['avg_objective']:<12.2f} {alg['avg_success_rate']:<12.2%} "
                  f"{alg['avg_time']:<10.3f}")
        print()

        # Instance-specific performance
        print("INSTANCE-SPECIFIC PERFORMANCE:")
        print("-" * 80)
        for instance in sorted(instances):
            print(f"\nInstance: {instance}")
            instance_perfs = [(alg, perf) for (alg, inst), perf in self.algorithm_performances.items()
                              if inst == instance]
            instance_perfs.sort(key=lambda x: x[1].avg_objective, reverse=True)

            for alg, perf in instance_perfs:
                feasible_str = f"({perf.feasibility_rate:.0%} feasible)" if perf.feasibility_rate < 1.0 else ""
                print(f"  {alg:<25}: {perf.avg_objective:>8.2f} ± {perf.std_objective:>6.2f} "
                      f"in {perf.avg_time:>6.3f}s {feasible_str}")

        self.print_cost_breakdown_summary()


def main():
    """Example usage of the metrics module."""
    analyzer = ResultsAnalyzer()

    # Load results
    analyzer.load_results()

    # Print summary report
    analyzer.print_summary_report()

    # Export to CSV
    analyzer.export_to_csv("results_summary.csv")
    analyzer.export_to_csv("results_raw.csv", include_raw_data=True)

    # Compare specific algorithms
    algorithms = list(set(alg for (alg, inst), _ in analyzer.algorithm_performances.items()))
    if len(algorithms) >= 2:
        comparisons = analyzer.compare_algorithms(algorithms[0], algorithms[1])

        print(f"\nSTATISTICAL COMPARISON: {algorithms[0]} vs {algorithms[1]}")
        print("-" * 80)
        for comp in comparisons:
            significance = "***" if comp.p_value < 0.001 else "**" if comp.p_value < 0.01 else "*" if comp.p_value < 0.05 else ""
            print(f"{comp.instance_name}: {comp.winner} wins (p={comp.p_value:.4f}){significance} "
                  f"[{comp.effect_size_interpretation} effect]")


if __name__ == "__main__":
    main()