import numpy as np
import random
import math
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum

from models.problem import CourseSchedulingProblem
from models.solution import Solution
from base import ImprovementAlgorithm

class LocalSearchMethod(Enum):
    """Methods for local search improvement."""
    BEST_IMPROVEMENT = "best_improvement"
    FIRST_IMPROVEMENT = "first_improvement"
    VARIABLE_DEPTH = "variable_depth"
    STEEPEST_DESCENT = "steepest_descent"


class VNSVariant(Enum):
    """Different VNS algorithm variants."""
    BASIC_VNS = "basic_vns"
    GENERAL_VNS = "general_vns"
    SKEWED_VNS = "skewed_vns"
    REDUCED_VNS = "reduced_vns"


class NeighborhoodStructure:
    """Represents a neighborhood structure with its operations."""

    def __init__(self, name: str, intensity: int, operation: Callable):
        self.name = name
        self.intensity = intensity  # 1 = small changes, 5 = large changes
        self.operation = operation

    def __str__(self):
        return f"N{self.intensity}({self.name})"


class VariableNeighborhoodSearch(ImprovementAlgorithm):
    """
    Variable Neighborhood Search algorithm for course scheduling optimization.

    Systematically changes neighborhood structures and uses local search
    to escape local optima and explore solution space efficiently.
    """

    def __init__(self, problem: CourseSchedulingProblem, **kwargs):
        """
        Initialize VNS algorithm.

        Args:
            problem: Problem instance
            **kwargs: Algorithm parameters
        """
        # Extract name from kwargs or use default
        algorithm_name = kwargs.pop('name', 'VariableNeighborhoodSearch')
        super().__init__(problem, name=algorithm_name, **kwargs)

        # VNS parameters
        self.k_max = kwargs.get('k_max', 5)
        self.vns_variant = VNSVariant(kwargs.get('vns_variant', 'basic_vns'))

        # Local search parameters
        self.local_search_method = LocalSearchMethod(kwargs.get('local_search_method', 'best_improvement'))
        self.local_search_iterations = kwargs.get('local_search_iterations', 50)

        # Shaking parameters
        self.shaking_intensity = kwargs.get('shaking_intensity', 1.0)
        self.adaptive_shaking = kwargs.get('adaptive_shaking', False)

        # Skewed VNS parameters (for accepting worse solutions)
        self.skewing_factor = kwargs.get('skewing_factor', 0.1)
        self.acceptance_probability = kwargs.get('acceptance_probability', 0.1)

        # Diversification parameters
        self.diversification_prob = kwargs.get('diversification_prob', 0.05)
        self.restart_threshold = kwargs.get('restart_threshold', 500)

        # Random seed
        self.random_seed = kwargs.get('random_seed', None)
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        # Initialize neighborhood structures
        self.neighborhoods = self._initialize_neighborhoods()

        # State tracking
        self.current_solution: Optional[Solution] = None
        self.current_fitness = float('-inf')
        self.current_k = 1
        self.iteration = 0
        self.stagnation_counter = 0
        self.local_search_calls = 0

        # Statistics
        self.neighborhood_usage = {i + 1: 0 for i in range(self.k_max)}
        self.improvements_per_neighborhood = {i + 1: 0 for i in range(self.k_max)}
        self.shaking_history = []
        self.local_search_improvements = 0

    def solve(self) -> Solution:
        """
        Main solving method implementing VNS.

        Returns:
            Best solution found
        """
        if self.verbose:
            print(f"Starting Variable Neighborhood Search:")
            print(f"  VNS variant: {self.vns_variant.value}")
            print(f"  k_max (neighborhoods): {self.k_max}")
            print(f"  Local search: {self.local_search_method.value}")
            print(f"  Shaking intensity: {self.shaking_intensity}")

        # Generate initial solution
        self.current_solution = self.generate_initial_solution()
        self.current_fitness = self._evaluate_solution(self.current_solution)
        self.stats.initial_objective = self.current_fitness

        # Apply initial local search
        self.current_solution = self._local_search(self.current_solution)
        self.current_fitness = self._evaluate_solution(self.current_solution)

        # Initialize best solution
        self.best_solution = self.current_solution.copy()
        best_fitness = self.current_fitness

        # Main VNS loop
        while self.iteration < self.max_iterations:
            k = 1

            # VNS neighborhood exploration
            while k <= self.k_max:

                # Shaking: generate random solution in k-th neighborhood
                shaken_solution = self._shaking(self.current_solution, k)

                # Local search improvement
                improved_solution = self._local_search(shaken_solution)
                improved_fitness = self._evaluate_solution(improved_solution)

                # Track neighborhood usage
                self.neighborhood_usage[k] += 1

                # Move or not decision
                if self._accept_solution(improved_fitness, self.current_fitness, k):
                    self.current_solution = improved_solution
                    self.current_fitness = improved_fitness

                    # Update best solution
                    if improved_fitness > best_fitness:
                        self.best_solution = improved_solution.copy()
                        best_fitness = improved_fitness
                        self.stats.improvements += 1
                        self.improvements_per_neighborhood[k] += 1
                        self.stagnation_counter = 0

                        if self.verbose:
                            print(f"    New best: {best_fitness:.2f} from N{k}")

                    # Restart from first neighborhood
                    k = 1

                    # Adaptive shaking adjustment
                    if self.adaptive_shaking:
                        self.shaking_intensity *= 0.95  # Reduce intensity after improvement
                else:
                    k += 1  # Try next neighborhood
                    self.stagnation_counter += 1

                    # Adaptive shaking adjustment
                    if self.adaptive_shaking and k > self.k_max * 0.7:
                        self.shaking_intensity *= 1.05  # Increase intensity when struggling

                # Check for diversification
                if (self.stagnation_counter > self.restart_threshold or
                        random.random() < self.diversification_prob):
                    self._diversify()
                    k = 1

                # Check termination
                if self.should_terminate(self.iteration):
                    break

            self.iteration += 1

            # Log progress
            self.log_progress(
                self.iteration,
                best_fitness,
                f"(current: {self.current_fitness:.2f}, k_used: {k - 1 if k > self.k_max else k})"
            )

        if self.verbose:
            print(f"\nVNS completed:")
            print(f"  Total iterations: {self.iteration}")
            print(f"  Local search calls: {self.local_search_calls}")
            print(f"  Local search improvements: {self.local_search_improvements}")
            self._print_neighborhood_statistics()
            self.best_solution.print_solution_summary()

        return self.best_solution

    def generate_initial_solution(self) -> Solution:
        """Generate initial solution using greedy approach."""
        solution = self.create_empty_solution()

        # Simple greedy assignment
        course_order = list(range(1, self.problem.N + 1))
        random.shuffle(course_order)

        for course_id in course_order:
            course_idx = course_id - 1
            duration = self.problem.course_durations[course_idx]
            max_participants = self.problem.course_max_participants[course_idx]
            participants = random.randint(max(1, max_participants // 2), max_participants)

            # Find compatible resources
            compatible_rooms = []
            for room_id in range(1, self.problem.M + 1):
                room_idx = room_id - 1
                if (self.problem.R_matrix[course_idx, room_idx] == 1 and
                        self.problem.room_capacities[room_idx] >= participants):
                    compatible_rooms.append(room_id)

            compatible_instructors = []
            for instructor_id in range(1, self.problem.K + 1):
                instructor_idx = instructor_id - 1
                if self.problem.I_matrix[instructor_idx, course_idx] == 1:
                    compatible_instructors.append(instructor_id)

            if not compatible_rooms or not compatible_instructors:
                continue

            # Try assignment
            for attempt in range(20):
                room_id = random.choice(compatible_rooms)
                instructor_id = random.choice(compatible_instructors)
                start_time = random.randint(1, max(1, self.problem.T - duration))

                # Check availability
                instructor_idx = instructor_id - 1
                available = True
                for t_offset in range(duration):
                    t = start_time + t_offset - 1
                    if t >= self.problem.T or self.problem.A_matrix[instructor_idx, t] == 0:
                        available = False
                        break

                if available and solution.assign_course(course_id, room_id, instructor_id,
                                                        start_time, duration, participants):
                    break

        return solution

    def improve_solution(self, solution: Solution) -> Solution:
        """Single improvement step using local search."""
        return self._local_search(solution)

    def _initialize_neighborhoods(self) -> List[NeighborhoodStructure]:
        """Initialize neighborhood structures ordered by intensity."""
        neighborhoods = []

        # N1: Single course swap (smallest change)
        neighborhoods.append(NeighborhoodStructure(
            "single_swap", 1, self._single_course_swap
        ))

        # N2: Course relocation (medium change)
        neighborhoods.append(NeighborhoodStructure(
            "relocate", 2, self._relocate_course
        ))

        # N3: Instructor change (medium change)
        neighborhoods.append(NeighborhoodStructure(
            "instructor_change", 2, self._change_instructor
        ))

        # N4: Multiple course swap (large change)
        neighborhoods.append(NeighborhoodStructure(
            "multi_swap", 3, self._multi_course_swap
        ))

        # N5: Time shift operations (large change)
        neighborhoods.append(NeighborhoodStructure(
            "time_shift", 3, self._time_shift_operations
        ))

        # N6: Add/Remove courses (very large change)
        if self.k_max >= 6:
            neighborhoods.append(NeighborhoodStructure(
                "add_remove", 4, self._add_remove_courses
            ))

        # N7: Massive perturbation (largest change)
        if self.k_max >= 7:
            neighborhoods.append(NeighborhoodStructure(
                "massive_perturbation", 5, self._massive_perturbation
            ))

        return neighborhoods[:self.k_max]

    def _shaking(self, solution: Solution, k: int) -> Solution:
        """Apply shaking in k-th neighborhood."""
        if k > len(self.neighborhoods):
            k = len(self.neighborhoods)

        neighborhood = self.neighborhoods[k - 1]
        shaken_solution = neighborhood.operation(solution, int(self.shaking_intensity))

        # Record shaking history
        self.shaking_history.append({
            'iteration': self.iteration,
            'neighborhood': k,
            'name': neighborhood.name,
            'intensity': self.shaking_intensity
        })

        return shaken_solution

    def _local_search(self, solution: Solution) -> Solution:
        """Apply local search to improve solution."""
        self.local_search_calls += 1
        current = solution.copy()
        current_fitness = self._evaluate_solution(current)

        if self.local_search_method == LocalSearchMethod.BEST_IMPROVEMENT:
            return self._best_improvement_search(current, current_fitness)
        elif self.local_search_method == LocalSearchMethod.FIRST_IMPROVEMENT:
            return self._first_improvement_search(current, current_fitness)
        elif self.local_search_method == LocalSearchMethod.VARIABLE_DEPTH:
            return self._variable_depth_search(current, current_fitness)
        else:  # STEEPEST_DESCENT
            return self._steepest_descent_search(current, current_fitness)

    def _best_improvement_search(self, solution: Solution, current_fitness: float) -> Solution:
        """Best improvement local search."""
        improved = True
        iterations = 0

        while improved and iterations < self.local_search_iterations:
            improved = False
            best_neighbor = None
            best_fitness = current_fitness

            # Generate all neighbors using simple operations
            neighbors = self._generate_simple_neighbors(solution)

            for neighbor in neighbors:
                neighbor_fitness = self._evaluate_solution(neighbor)
                if neighbor_fitness > best_fitness:
                    best_neighbor = neighbor
                    best_fitness = neighbor_fitness
                    improved = True

            if improved:
                solution = best_neighbor
                current_fitness = best_fitness
                self.local_search_improvements += 1

            iterations += 1

        return solution

    def _first_improvement_search(self, solution: Solution, current_fitness: float) -> Solution:
        """First improvement local search."""
        iterations = 0

        while iterations < self.local_search_iterations:
            # Generate neighbors one by one
            neighbors = self._generate_simple_neighbors(solution)

            improved = False
            for neighbor in neighbors:
                neighbor_fitness = self._evaluate_solution(neighbor)
                if neighbor_fitness > current_fitness:
                    solution = neighbor
                    current_fitness = neighbor_fitness
                    self.local_search_improvements += 1
                    improved = True
                    break

            if not improved:
                break

            iterations += 1

        return solution

    def _variable_depth_search(self, solution: Solution, current_fitness: float) -> Solution:
        """Variable depth local search with different intensities."""
        # Try different search depths
        for depth in [1, 2, 3]:
            for _ in range(min(self.local_search_iterations // depth, 20)):
                # Apply multiple moves of decreasing intensity
                temp_solution = solution.copy()
                temp_fitness = current_fitness

                for d in range(depth):
                    neighbors = self._generate_simple_neighbors(temp_solution)
                    if neighbors:
                        neighbor = random.choice(neighbors)
                        neighbor_fitness = self._evaluate_solution(neighbor)

                        if neighbor_fitness > temp_fitness or d == 0:  # Always accept first move
                            temp_solution = neighbor
                            temp_fitness = neighbor_fitness

                # Accept if overall improvement
                if temp_fitness > current_fitness:
                    solution = temp_solution
                    current_fitness = temp_fitness
                    self.local_search_improvements += 1

        return solution

    def _steepest_descent_search(self, solution: Solution, current_fitness: float) -> Solution:
        """Steepest descent local search."""
        return self._best_improvement_search(solution, current_fitness)  # Same as best improvement

    def _generate_simple_neighbors(self, solution: Solution) -> List[Solution]:
        """Generate simple neighbors for local search."""
        neighbors = []
        assignments = solution.get_all_assignments()

        # Limit number of neighbors to avoid excessive computation
        max_neighbors = 20
        neighbor_count = 0

        # Swap neighbors
        for i, assignment1 in enumerate(assignments):
            if neighbor_count >= max_neighbors:
                break
            for j, assignment2 in enumerate(assignments[i + 1:], i + 1):
                if neighbor_count >= max_neighbors:
                    break

                neighbor = solution.copy()
                if neighbor.swap_courses(assignment1['course_id'], assignment2['course_id']):
                    neighbors.append(neighbor)
                    neighbor_count += 1

        # Time shift neighbors
        for assignment in assignments[:5]:  # Limit to first 5
            if neighbor_count >= max_neighbors:
                break

            course_id = assignment['course_id']
            duration = assignment['duration']

            for new_time in [assignment['start_time'] - 1, assignment['start_time'] + 1]:
                if neighbor_count >= max_neighbors:
                    break
                if new_time >= 1 and new_time + duration - 1 <= self.problem.T:
                    neighbor = solution.copy()
                    neighbor.remove_course_assignment(course_id)
                    if neighbor.assign_course(
                            course_id, assignment['room_id'], assignment['instructor_id'],
                            new_time, duration, assignment['participants']
                    ):
                        neighbors.append(neighbor)
                        neighbor_count += 1

        return neighbors

    def _evaluate_solution(self, solution: Solution) -> float:
        """Evaluate solution fitness with penalties."""
        x, y, u = solution.to_arrays()
        objective = self.problem.calculate_objective(x, y, u)

        # Add penalties for infeasibility
        is_feasible, violations = self.problem.validate_solution(x, y, u)
        if not is_feasible:
            penalty = len(violations) * 200
            fitness = objective - penalty
        else:
            fitness = objective

        self.stats.evaluations += 1
        return fitness

    def _accept_solution(self, new_fitness: float, current_fitness: float, k: int) -> bool:
        """Decide whether to accept new solution."""
        if self.vns_variant == VNSVariant.BASIC_VNS:
            # Basic VNS: accept only improvements
            return new_fitness > current_fitness

        elif self.vns_variant == VNSVariant.SKEWED_VNS:
            # Skewed VNS: accept worse solutions with small probability
            if new_fitness > current_fitness:
                return True
            else:
                # Accept worse solution with probability based on difference and k
                diff = current_fitness - new_fitness
                probability = self.acceptance_probability * math.exp(-diff / (self.skewing_factor * k))
                return random.random() < probability

        else:  # GENERAL_VNS, REDUCED_VNS
            # Accept improvements and some worse solutions
            if new_fitness > current_fitness:
                return True
            elif k > self.k_max * 0.6:  # Accept worse solutions for larger k
                return random.random() < 0.05
            else:
                return False

    def _diversify(self) -> None:
        """Perform diversification when stagnating."""
        if self.verbose:
            print(f"  Diversifying solution...")

        # Apply massive perturbation
        self.current_solution = self._massive_perturbation(self.current_solution, 3)
        self.current_fitness = self._evaluate_solution(self.current_solution)

        # Reset counters
        self.stagnation_counter = 0

        # Reset adaptive shaking intensity
        if self.adaptive_shaking:
            self.shaking_intensity = 1.0

    def _print_neighborhood_statistics(self) -> None:
        """Print statistics about neighborhood usage."""
        print(f"\nNeighborhood usage statistics:")
        for k in range(1, self.k_max + 1):
            if k <= len(self.neighborhoods):
                name = self.neighborhoods[k - 1].name
                usage = self.neighborhood_usage[k]
                improvements = self.improvements_per_neighborhood[k]
                success_rate = (improvements / usage * 100) if usage > 0 else 0
                print(f"  N{k} ({name}): {usage} uses, {improvements} improvements ({success_rate:.1f}%)")

    # Neighborhood operation methods
    def _single_course_swap(self, solution: Solution, intensity: int) -> Solution:
        """Swap assignments of two courses."""
        neighbor = solution.copy()
        assignments = neighbor.get_all_assignments()

        if len(assignments) >= 2:
            for _ in range(intensity):
                course1, course2 = random.sample(assignments, 2)
                neighbor.swap_courses(course1['course_id'], course2['course_id'])

        return neighbor

    def _relocate_course(self, solution: Solution, intensity: int) -> Solution:
        """Relocate courses to different rooms/times."""
        neighbor = solution.copy()
        assignments = neighbor.get_all_assignments()

        for _ in range(min(intensity, len(assignments))):
            assignment = random.choice(assignments)
            course_id = assignment['course_id']
            course_idx = course_id - 1

            # Remove current assignment
            neighbor.remove_course_assignment(course_id)

            # Try new assignment
            duration = self.problem.course_durations[course_idx]
            participants = assignment['participants']

            for attempt in range(10):
                # Find compatible room
                compatible_rooms = []
                for room_id in range(1, self.problem.M + 1):
                    room_idx = room_id - 1
                    if (self.problem.R_matrix[course_idx, room_idx] == 1 and
                            self.problem.room_capacities[room_idx] >= participants):
                        compatible_rooms.append(room_id)

                if not compatible_rooms:
                    break

                room_id = random.choice(compatible_rooms)
                instructor_id = assignment['instructor_id']  # Keep same instructor
                start_time = random.randint(1, max(1, self.problem.T - duration))

                if neighbor.assign_course(course_id, room_id, instructor_id,
                                          start_time, duration, participants):
                    break
            else:
                # Restore original if failed
                neighbor.assign_course(
                    course_id, assignment['room_id'], assignment['instructor_id'],
                    assignment['start_time'], assignment['duration'], participants
                )

        return neighbor

    def _change_instructor(self, solution: Solution, intensity: int) -> Solution:
        """Change instructors for courses."""
        neighbor = solution.copy()
        assignments = neighbor.get_all_assignments()

        for _ in range(min(intensity, len(assignments))):
            assignment = random.choice(assignments)
            course_id = assignment['course_id']
            course_idx = course_id - 1

            # Find alternative instructors
            compatible_instructors = []
            for instructor_id in range(1, self.problem.K + 1):
                instructor_idx = instructor_id - 1
                if (self.problem.I_matrix[instructor_idx, course_idx] == 1 and
                        instructor_id != assignment['instructor_id']):
                    compatible_instructors.append(instructor_id)

            if compatible_instructors:
                neighbor.remove_course_assignment(course_id)
                new_instructor = random.choice(compatible_instructors)

                success = neighbor.assign_course(
                    course_id, assignment['room_id'], new_instructor,
                    assignment['start_time'], assignment['duration'],
                    assignment['participants']
                )

                if not success:
                    # Restore original
                    neighbor.assign_course(
                        course_id, assignment['room_id'], assignment['instructor_id'],
                        assignment['start_time'], assignment['duration'],
                        assignment['participants']
                    )

        return neighbor

    def _multi_course_swap(self, solution: Solution, intensity: int) -> Solution:
        """Swap multiple pairs of courses."""
        neighbor = solution.copy()
        assignments = neighbor.get_all_assignments()

        for _ in range(min(intensity, len(assignments) // 2)):
            if len(assignments) >= 4:
                # Select 4 courses and swap in pairs
                selected = random.sample(assignments, 4)
                neighbor.swap_courses(selected[0]['course_id'], selected[1]['course_id'])
                neighbor.swap_courses(selected[2]['course_id'], selected[3]['course_id'])

        return neighbor

    def _time_shift_operations(self, solution: Solution, intensity: int) -> Solution:
        """Perform multiple time shift operations."""
        neighbor = solution.copy()
        assignments = neighbor.get_all_assignments()

        for _ in range(min(intensity * 2, len(assignments))):
            assignment = random.choice(assignments)
            course_id = assignment['course_id']
            duration = assignment['duration']

            # Try shifting time
            shift = random.choice([-2, -1, 1, 2])
            new_time = assignment['start_time'] + shift

            if new_time >= 1 and new_time + duration - 1 <= self.problem.T:
                neighbor.remove_course_assignment(course_id)
                success = neighbor.assign_course(
                    course_id, assignment['room_id'], assignment['instructor_id'],
                    new_time, duration, assignment['participants']
                )

                if not success:
                    # Restore original
                    neighbor.assign_course(
                        course_id, assignment['room_id'], assignment['instructor_id'],
                        assignment['start_time'], duration, assignment['participants']
                    )

        return neighbor

    def _add_remove_courses(self, solution: Solution, intensity: int) -> Solution:
        """Add and remove courses."""
        neighbor = solution.copy()

        for _ in range(intensity):
            if random.random() < 0.5:
                # Try to add unscheduled course
                unscheduled = neighbor.get_unscheduled_courses()
                if unscheduled:
                    course_id = random.choice(list(unscheduled))
                    course_idx = course_id - 1
                    duration = self.problem.course_durations[course_idx]
                    participants = self.problem.course_max_participants[course_idx]

                    # Try random assignment
                    for attempt in range(5):
                        room_id = random.randint(1, self.problem.M)
                        instructor_id = random.randint(1, self.problem.K)
                        start_time = random.randint(1, max(1, self.problem.T - duration))

                        if neighbor.assign_course(course_id, room_id, instructor_id,
                                                  start_time, duration, participants):
                            break
            else:
                # Remove random course
                assignments = neighbor.get_all_assignments()
                if assignments:
                    assignment = random.choice(assignments)
                    neighbor.remove_course_assignment(assignment['course_id'])

        return neighbor

    def _massive_perturbation(self, solution: Solution, intensity: int) -> Solution:
        """Apply massive perturbation with multiple operations."""
        neighbor = solution.copy()

        # Apply multiple different operations
        for _ in range(intensity):
            operation = random.choice([
                self._single_course_swap,
                self._relocate_course,
                self._change_instructor,
                self._time_shift_operations
            ])
            neighbor = operation(neighbor, 1)

        return neighbor

def create_vns_variants() -> List[Dict]:
    """
    Create different VNS variants with various parameter combinations.

    Returns:
        List of parameter dictionaries for different VNS variants
    """
    variants = []

    # Basic VNS
    variants.append({
        'name': 'VNS_Basic',
        'k_max': 5,
        'vns_variant': 'basic_vns',
        'local_search_method': 'best_improvement',
        'local_search_iterations': 50,
        'shaking_intensity': 1.0,
        'max_iterations': 1000
    })

    # General VNS with variable depth search
    variants.append({
        'name': 'VNS_General',
        'k_max': 6,
        'vns_variant': 'general_vns',
        'local_search_method': 'variable_depth',
        'local_search_iterations': 75,
        'shaking_intensity': 1.2,
        'adaptive_shaking': True,
        'max_iterations': 1200
    })

    # Skewed VNS (accepts worse solutions)
    variants.append({
        'name': 'VNS_Skewed',
        'k_max': 5,
        'vns_variant': 'skewed_vns',
        'local_search_method': 'first_improvement',
        'local_search_iterations': 40,
        'shaking_intensity': 0.8,
        'skewing_factor': 0.1,
        'acceptance_probability': 0.05,
        'max_iterations': 1500
    })

    # Fast VNS with reduced neighborhoods
    variants.append({
        'name': 'VNS_Fast',
        'k_max': 4,
        'vns_variant': 'reduced_vns',
        'local_search_method': 'first_improvement',
        'local_search_iterations': 25,
        'shaking_intensity': 1.5,
        'diversification_prob': 0.1,
        'max_iterations': 800
    })

    # Intensive VNS with deep search
    variants.append({
        'name': 'VNS_Intensive',
        'k_max': 7,
        'vns_variant': 'general_vns',
        'local_search_method': 'steepest_descent',
        'local_search_iterations': 100,
        'shaking_intensity': 1.0,
        'adaptive_shaking': True,
        'restart_threshold': 300,
        'max_iterations': 2000
    })

    return variants

if __name__ == "__main__":
    # Test VNS algorithm
    import sys
    import os
    import math

    # Import problem
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from utils.data_loader import load_instance

    try:
        # Load test instance
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        instance_path = os.path.join(project_root, "data", "instances", "small", "small_01")
        loader = load_instance(instance_path)
        problem = CourseSchedulingProblem(loader)

        print("Testing Variable Neighborhood Search variants...")

        # Test different variants
        variants = create_vns_variants()

        for variant in variants[:2]:  # Test first 2 variants
            print(f"\n{'=' * 60}")
            print(f"Testing: {variant['name']}")
            print(f"{'=' * 60}")

            # Create and run algorithm
            vns = VariableNeighborhoodSearch(problem, verbose=True, **variant)
            solution = vns.run()

            # Print results
            vns.print_statistics()

            # Validate solution
            x, y, u = solution.to_arrays()
            is_feasible, violations = problem.validate_solution(x, y, u)

            print(f"\nSolution validation:")
            print(f"  Feasible: {is_feasible}")
            if violations:
                print(f"  Violations: {len(violations)}")
                for violation in violations[:3]:  # Show first 3
                    print(f"    - {violation}")

            # Print VNS-specific stats
            print(f"\nVNS statistics:")
            print(f"  Neighborhoods defined: {len(vns.neighborhoods)}")
            print(f"  Shaking operations: {len(vns.shaking_history)}")
            if vns.shaking_history:
                last_shaking = vns.shaking_history[-1]
                print(f"  Final shaking intensity: {last_shaking['intensity']:.2f}")

        print(f"\n✓ Variable Neighborhood Search testing completed!")

    except Exception as e:
        print(f"Error testing VNS: {e}")
        import traceback
        traceback.print_exc()