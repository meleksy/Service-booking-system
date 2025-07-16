import time
import numpy as np
import json
import os
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import sys
import os

# Import problem and solution classes
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from models.problem import CourseSchedulingProblem
from models.solution import Solution


@dataclass
class AlgorithmStatistics:
    """Statistics collected during algorithm execution."""

    # Timing information
    execution_time: float = 0.0
    setup_time: float = 0.0
    solve_time: float = 0.0

    # Solution quality
    best_objective: Optional[float] = None
    final_objective: Optional[float] = None
    initial_objective: Optional[float] = None

    # Algorithm progress
    iterations: int = 0
    evaluations: int = 0
    improvements: int = 0

    # Feasibility tracking
    feasible_solutions_found: int = 0
    infeasible_solutions_generated: int = 0

    # Convergence data (for plotting)
    objective_history: List[float] = None
    time_history: List[float] = None

    # Algorithm-specific metrics
    custom_metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.objective_history is None:
            self.objective_history = []
        if self.time_history is None:
            self.time_history = []
        if self.custom_metrics is None:
            self.custom_metrics = {}


class Algorithm(ABC):
    """
    Abstract base class for all optimization algorithms.

    Provides common interface and functionality for course scheduling algorithms.
    All specific algorithms (greedy, genetic, etc.) should inherit from this class.
    """

    def __init__(self, problem: CourseSchedulingProblem, name: str = "BaseAlgorithm", **kwargs):
        """
        Initialize algorithm with problem instance.

        Args:
            problem: CourseSchedulingProblem instance
            name: Algorithm name for identification
            **kwargs: Algorithm-specific parameters
        """
        self.problem = problem
        self.name = name
        self.parameters = kwargs

        # Solution tracking
        self.best_solution: Optional[Solution] = None
        self.current_solution: Optional[Solution] = None
        self.initial_solution: Optional[Solution] = None

        # Statistics
        self.stats = AlgorithmStatistics()

        # State tracking
        self.is_solved = False
        self.start_time = 0.0

        # Logging
        self.verbose = kwargs.get('verbose', False)
        self.log_frequency = kwargs.get('log_frequency', 10)

        # Results saving
        self.save_results = kwargs.get('save_results', True)
        # Default path relative to project root (2 levels up from src/algorithms/)
        default_results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'results')
        self.results_dir = kwargs.get('results_dir', default_results_dir)

        # Auto-detect instance name from problem if available
        if hasattr(problem, 'instance_name'):
            default_instance_name = problem.instance_name
        else:
            default_instance_name = 'unknown_instance'

        self.instance_name = kwargs.get('instance_name', default_instance_name)
        self.run_id = kwargs.get('run_id', datetime.now().strftime('%Y%m%d_%H%M%S'))

        # Create results directory if it doesn't exist
        if self.save_results:
            os.makedirs(self.results_dir, exist_ok=True)

    @abstractmethod
    def solve(self) -> Solution:
        """
        Main solving method - must be implemented by subclasses.

        Returns:
            Best solution found
        """
        pass

    def initialize(self) -> None:
        """
        Initialize algorithm state before solving.
        Can be overridden by subclasses for custom initialization.
        """
        self.stats = AlgorithmStatistics()
        self.best_solution = None
        self.current_solution = None
        self.is_solved = False

        if self.verbose:
            print(f"Initializing {self.name} algorithm...")
            print(f"Problem: {self.problem.N} courses, {self.problem.M} rooms, "
                  f"{self.problem.K} instructors, {self.problem.T} time slots")

    def finalize(self) -> None:
        """
        Finalize algorithm execution and compute final statistics.
        Can be overridden by subclasses for custom cleanup.
        """
        self.stats.execution_time = time.time() - self.start_time
        self.is_solved = True

        if self.best_solution:
            x, y, u = self.best_solution.to_arrays()
            self.stats.best_objective = self.problem.calculate_objective(x, y, u)
            self.stats.final_objective = self.stats.best_objective

        # Save results automatically
        if self.save_results:
            self._save_results()

        if self.verbose:
            print(f"\n{self.name} completed:")
            print(f"  Execution time: {self.stats.execution_time:.2f}s")
            if self.stats.best_objective is not None:
                print(f"  Best objective: {self.stats.best_objective:.2f}")
            else:
                print(f"  Best objective: None (no solution found)")
            print(f"  Iterations: {self.stats.iterations}")
            print(f"  Evaluations: {self.stats.evaluations}")
            if self.save_results:
                print(f"  Results saved to: {self.results_dir}")

    def run(self) -> Solution:
        """
        Run the complete algorithm with timing and statistics tracking.

        Returns:
            Best solution found
        """
        self.start_time = time.time()

        # Initialize
        setup_start = time.time()
        self.initialize()
        self.stats.setup_time = time.time() - setup_start

        # Solve
        solve_start = time.time()
        solution = self.solve()
        self.stats.solve_time = time.time() - solve_start

        # Finalize
        self.finalize()

        return solution

    def evaluate_solution(self, solution: Solution) -> float:
        """
        Evaluate solution and update statistics.

        Args:
            solution: Solution to evaluate

        Returns:
            Objective value
        """
        x, y, u = solution.to_arrays()

        # Check feasibility
        is_feasible = self.problem.is_feasible(x, y, u)
        if is_feasible:
            self.stats.feasible_solutions_found += 1
        else:
            self.stats.infeasible_solutions_generated += 1

        # Calculate objective (even for infeasible solutions)
        objective = self.problem.calculate_objective(x, y, u)

        # Update statistics
        self.stats.evaluations += 1

        # Update best solution if this is better and feasible
        if is_feasible and (self.best_solution is None or
                            objective > self.stats.best_objective):
            self.best_solution = solution.copy()
            self.stats.best_objective = objective
            self.stats.improvements += 1

            if self.verbose:
                print(f"  New best solution found: {objective:.2f}")

        return objective

    def log_progress(self, iteration: int, objective: float, additional_info: str = "") -> None:
        """
        Log algorithm progress.

        Args:
            iteration: Current iteration number
            objective: Current objective value
            additional_info: Additional information to log
        """
        self.stats.iterations = iteration

        # Record convergence data
        current_time = time.time() - self.start_time
        self.stats.objective_history.append(objective)
        self.stats.time_history.append(current_time)

        # Print progress if verbose and at logging frequency
        if self.verbose and (iteration % self.log_frequency == 0 or iteration == 1):
            print(f"  Iteration {iteration}: objective = {objective:.2f} {additional_info}")

    def create_empty_solution(self) -> Solution:
        """Create empty solution with problem dimensions."""
        return Solution(self.problem.N, self.problem.M, self.problem.K, self.problem.T, self.problem)

    def create_random_solution(self, seed: Optional[int] = None) -> Solution:
        """
        Create random feasible solution (basic implementation).
        Can be overridden by subclasses for better random generation.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Random solution (may be infeasible)
        """
        if seed is not None:
            np.random.seed(seed)

        solution = self.create_empty_solution()

        # Try to assign each course randomly
        for course_id in range(1, self.problem.N + 1):
            # Get random number of participants (up to max)
            max_participants = self.problem.course_max_participants[course_id - 1]
            participants = np.random.randint(1, max_participants + 1)

            # Get duration
            duration = self.problem.course_durations[course_id - 1]

            # Try random assignments (with limited attempts)
            for attempt in range(50):  # Max 50 attempts per course
                # Random room, instructor, start time
                room_id = np.random.randint(1, self.problem.M + 1)
                instructor_id = np.random.randint(1, self.problem.K + 1)
                start_time = np.random.randint(1, self.problem.T - duration + 2)

                # Try assignment
                if solution.assign_course(course_id, room_id, instructor_id,
                                          start_time, duration, participants):
                    break  # Success, move to next course

        return solution

    def get_solution(self) -> Optional[Solution]:
        """Get the best solution found."""
        return self.best_solution

    def get_statistics(self) -> AlgorithmStatistics:
        """Get algorithm execution statistics."""
        return self.stats

    def set_parameter(self, key: str, value: Any) -> None:
        """Set algorithm parameter."""
        self.parameters[key] = value

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get algorithm parameter."""
        return self.parameters.get(key, default)

    def print_statistics(self) -> None:
        """Print detailed algorithm statistics."""
        print(f"\n=== {self.name} Statistics ===")
        print(f"Execution time: {self.stats.execution_time:.3f}s")
        print(f"  Setup time: {self.stats.setup_time:.3f}s")
        print(f"  Solve time: {self.stats.solve_time:.3f}s")

        print(f"\nSolution quality:")
        if self.stats.best_objective is not None:
            print(f"  Best objective: {self.stats.best_objective:.2f}")
        else:
            print(f"  Best objective: None")

        if self.stats.initial_objective:
            improvement = (self.stats.best_objective or 0) - self.stats.initial_objective
            print(f"  Initial objective: {self.stats.initial_objective:.2f}")
            print(f"  Improvement: {improvement:.2f}")

        print(f"\nAlgorithm progress:")
        print(f"  Iterations: {self.stats.iterations}")
        print(f"  Evaluations: {self.stats.evaluations}")
        print(f"  Improvements: {self.stats.improvements}")

        print(f"\nFeasibility:")
        print(f"  Feasible solutions: {self.stats.feasible_solutions_found}")
        print(f"  Infeasible solutions: {self.stats.infeasible_solutions_generated}")

        if self.stats.custom_metrics:
            print(f"\nCustom metrics:")
            for key, value in self.stats.custom_metrics.items():
                print(f"  {key}: {value}")

    def _save_results(self) -> None:
        """Save algorithm results to JSON file."""
        try:
            # Prepare results data
            results = {
                'metadata': {
                    'algorithm': self.name,
                    'instance_name': self.instance_name,
                    'run_id': self.run_id,
                    'timestamp': datetime.now().isoformat(),
                    'parameters': self.parameters
                },
                'problem_info': {
                    'num_courses': self.problem.N,
                    'num_rooms': self.problem.M,
                    'num_instructors': self.problem.K,
                    'time_horizon': self.problem.T
                },
                'statistics': asdict(self.stats),
                'solution': None
            }

            # Add solution data if available
            if self.best_solution:
                x, y, u = self.best_solution.to_arrays()
                is_feasible, violations = self.problem.validate_solution(x, y, u)

                # Get course assignments
                assignments = self.best_solution.get_all_assignments()

                # Calculate utilization
                utilization = self.best_solution.calculate_utilization()

                detailed_objective = self.problem.calculate_detailed_objective(x, y, u)

                results['solution'] = {
                    'objective_value': self.stats.best_objective,
                    'is_feasible': is_feasible,
                    'num_violations': len(violations) if violations else 0,
                    'course_assignments': assignments,
                    'utilization': utilization,
                    'scheduled_courses': len(assignments),
                    'total_courses': self.problem.N
                }

                # Add violations (first 10 for brevity)
                if violations:
                    results['solution']['violations'] = violations[:10]

            # Generate filename
            filename = f"{self.name}_{self.instance_name}_{self.run_id}.json"
            filepath = os.path.join(self.results_dir, filename)

            # Save to file
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            if self.verbose:
                print(f"  Results saved to: {filepath}")

        except Exception as e:
            if self.verbose:
                print(f"  Warning: Failed to save results - {e}")

    def load_results(self, filepath: str) -> Dict:
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)

    def set_instance_info(self, instance_name: str, run_id: Optional[str] = None) -> None:
        """Set instance information for result saving."""
        self.instance_name = instance_name
        if run_id:
            self.run_id = run_id
        """
        Save convergence data to file for plotting.

        Args:
            filename: Output filename
        """
        import json

        data = {
            'algorithm': self.name,
            'parameters': self.parameters,
            'objective_history': self.stats.objective_history,
            'time_history': self.stats.time_history,
            'final_stats': {
                'execution_time': self.stats.execution_time,
                'best_objective': self.stats.best_objective,
                'iterations': self.stats.iterations,
                'evaluations': self.stats.evaluations
            }
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        if self.verbose:
            print(f"Convergence data saved to {filename}")


class ConstructiveAlgorithm(Algorithm):
    """
    Base class for constructive algorithms (e.g., greedy).
    Builds solution from scratch step by step.
    """

    def __init__(self, problem: CourseSchedulingProblem, **kwargs):
        # Extract name or use default
        algorithm_name = kwargs.pop('name', 'ConstructiveAlgorithm')
        super().__init__(problem, name=algorithm_name, **kwargs)

    @abstractmethod
    def select_next_course(self, solution: Solution) -> Optional[int]:
        """Select next course to assign."""
        pass

    @abstractmethod
    def select_assignment(self, solution: Solution, course_id: int) -> Optional[Dict]:
        """Select room, instructor, time for given course."""
        pass


class ImprovementAlgorithm(Algorithm):
    """
    Base class for improvement algorithms (e.g., local search, metaheuristics).
    Starts with initial solution and improves it iteratively.
    """

    def __init__(self, problem: CourseSchedulingProblem, **kwargs):
        # Extract name or use default
        algorithm_name = kwargs.pop('name', 'ImprovementAlgorithm')
        super().__init__(problem, name=algorithm_name, **kwargs)
        self.max_iterations = kwargs.get('max_iterations', 1000)
        self.max_time = kwargs.get('max_time', 300)  # 5 minutes default

    @abstractmethod
    def generate_initial_solution(self) -> Solution:
        """Generate initial solution."""
        pass

    @abstractmethod
    def improve_solution(self, solution: Solution) -> Solution:
        """Improve given solution (one iteration)."""
        pass

    def should_terminate(self, iteration: int) -> bool:
        """Check termination criteria."""
        if iteration >= self.max_iterations:
            return True

        if time.time() - self.start_time >= self.max_time:
            return True

        return False


if __name__ == "__main__":
    # Example usage - test base algorithm functionality
    print("Testing base algorithm classes...")

    # This would normally be done with a real algorithm implementation
    # For now, just test the base class functionality

    # Import problem
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from utils.data_loader import load_instance

    try:
        # Load test instance
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        instance_path = os.path.join(project_root, "data", "instances", "small", "small_01")
        loader = load_instance(instance_path)
        problem = CourseSchedulingProblem(loader)


        # Create a mock algorithm for testing
        class MockAlgorithm(Algorithm):
            def solve(self) -> Solution:
                # Simple mock: create empty solution
                solution = self.create_empty_solution()

                # Try to assign one course for testing
                success = solution.assign_course(1, 1, 1, 1, 2, 10)
                if success:
                    self.evaluate_solution(solution)

                return solution


        # Test mock algorithm
        mock_algo = MockAlgorithm(problem, verbose=True)
        solution = mock_algo.run()
        mock_algo.print_statistics()

        print("\n✓ Base algorithm classes working correctly!")

    except Exception as e:
        print(f"Error testing base classes: {e}")
        print("Make sure test instance is available")