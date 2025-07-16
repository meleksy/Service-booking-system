import numpy as np
import random
import math
from typing import List, Dict, Tuple, Optional
from enum import Enum

from models.problem import CourseSchedulingProblem
from models.solution import Solution
from base import ImprovementAlgorithm


class CoolingSchedule(Enum):
    """Cooling schedule methods for temperature reduction."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    ADAPTIVE = "adaptive"


class NeighborhoodMethod(Enum):
    """Methods for generating neighboring solutions."""
    SWAP_COURSES = "swap_courses"
    RELOCATE_COURSE = "relocate_course"
    TIME_SHIFT = "time_shift"
    INSTRUCTOR_CHANGE = "instructor_change"
    ADD_REMOVE = "add_remove"
    MIXED = "mixed"


class SimulatedAnnealingAlgorithm(ImprovementAlgorithm):
    """
    Simulated Annealing algorithm for course scheduling optimization.

    Uses temperature-controlled acceptance of worse solutions to escape
    local optima and find better global solutions.
    """

    def __init__(self, problem: CourseSchedulingProblem, **kwargs):
        """
        Initialize simulated annealing algorithm.

        Args:
            problem: Problem instance
            **kwargs: Algorithm parameters
        """
        # Extract name from kwargs or use default
        algorithm_name = kwargs.pop('name', 'SimulatedAnnealingAlgorithm')
        super().__init__(problem, name=algorithm_name, **kwargs)

        # Temperature parameters
        self.initial_temperature = kwargs.get('initial_temperature', 1000.0)
        self.final_temperature = kwargs.get('final_temperature', 0.1)
        self.cooling_rate = kwargs.get('cooling_rate', 0.95)

        # Iteration parameters
        self.iterations_per_temp = kwargs.get('iterations_per_temp', 50)
        self.max_iterations = kwargs.get('max_iterations', 10000)

        # Cooling schedule
        self.cooling_schedule = CoolingSchedule(kwargs.get('cooling_schedule', 'exponential'))

        # Neighborhood parameters
        self.neighborhood_method = NeighborhoodMethod(kwargs.get('neighborhood_method', 'mixed'))
        self.neighborhood_size = kwargs.get('neighborhood_size', 1)

        # Adaptive parameters
        self.adaptive_cooling = kwargs.get('adaptive_cooling', False)
        self.acceptance_rate_target = kwargs.get('acceptance_rate_target', 0.4)
        self.temperature_adjustment_factor = kwargs.get('temperature_adjustment_factor', 1.1)

        # Restart parameters
        self.restart_threshold = kwargs.get('restart_threshold', 1000)
        self.max_restarts = kwargs.get('max_restarts', 3)

        # Random seed
        self.random_seed = kwargs.get('random_seed', None)
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        # State tracking
        self.current_temperature = self.initial_temperature
        self.current_solution: Optional[Solution] = None
        self.current_fitness = float('-inf')
        self.iteration = 0
        self.temp_iteration = 0
        self.restarts_used = 0
        self.stagnation_counter = 0

        # Statistics
        self.temperature_history = []
        self.acceptance_history = []
        self.accepted_worse = 0
        self.total_moves = 0

    def solve(self) -> Solution:
        """
        Main solving method implementing simulated annealing.

        Returns:
            Best solution found
        """
        if self.verbose:
            print(f"Starting simulated annealing:")
            print(f"  Initial temperature: {self.initial_temperature}")
            print(f"  Final temperature: {self.final_temperature}")
            print(f"  Cooling rate: {self.cooling_rate}")
            print(f"  Cooling schedule: {self.cooling_schedule.value}")
            print(f"  Neighborhood method: {self.neighborhood_method.value}")
            print(f"  Iterations per temperature: {self.iterations_per_temp}")

        # Generate initial solution
        self.current_solution = self.generate_initial_solution()
        self.current_fitness = self._evaluate_solution(self.current_solution)
        self.stats.initial_objective = self.current_fitness

        # Set best solution
        self.best_solution = self.current_solution.copy()
        best_fitness = self.current_fitness

        # Main annealing loop
        while (self.current_temperature > self.final_temperature and
               self.iteration < self.max_iterations):

            # Inner loop at current temperature
            accepted_at_temp = 0

            for temp_iter in range(self.iterations_per_temp):
                self.temp_iteration = temp_iter

                # Generate neighbor
                neighbor = self._generate_neighbor(self.current_solution)
                neighbor_fitness = self._evaluate_solution(neighbor)

                # Calculate fitness difference
                delta_E = neighbor_fitness - self.current_fitness

                # Acceptance decision
                if self._accept_move(delta_E, self.current_temperature):
                    self.current_solution = neighbor
                    self.current_fitness = neighbor_fitness
                    accepted_at_temp += 1

                    if delta_E < 0:  # Accepted worse solution
                        self.accepted_worse += 1

                    # Update best solution if better
                    if neighbor_fitness > best_fitness:
                        self.best_solution = neighbor.copy()
                        best_fitness = neighbor_fitness
                        self.stats.improvements += 1
                        self.stagnation_counter = 0

                        if self.verbose:
                            print(f"    New best: {best_fitness:.2f} at T={self.current_temperature:.2f}")
                    else:
                        self.stagnation_counter += 1

                self.total_moves += 1
                self.iteration += 1

                # Check early termination
                if self.should_terminate(self.iteration):
                    break

            # Calculate acceptance rate at this temperature
            acceptance_rate = accepted_at_temp / self.iterations_per_temp
            self.acceptance_history.append(acceptance_rate)

            # Adaptive cooling adjustment
            if self.adaptive_cooling:
                self._adjust_temperature_adaptive(acceptance_rate)
            else:
                # Standard cooling
                self.current_temperature = self._cool_temperature(self.current_temperature)

            # Record temperature
            self.temperature_history.append(self.current_temperature)

            # Log progress
            self.log_progress(
                self.iteration,
                best_fitness,
                f"(T={self.current_temperature:.2f}, acc_rate={acceptance_rate:.2f})"
            )

            # Check for restart
            if (self.stagnation_counter > self.restart_threshold and
                    self.restarts_used < self.max_restarts):
                self._restart()

        if self.verbose:
            print(f"\nSimulated annealing completed:")
            print(f"  Final temperature: {self.current_temperature:.4f}")
            print(f"  Total iterations: {self.iteration}")
            print(f"  Restarts used: {self.restarts_used}")
            print(f"  Acceptance rate: {self.accepted_worse / max(1, self.total_moves):.2%}")
            self.best_solution.print_solution_summary()

        return self.best_solution

    def generate_initial_solution(self) -> Solution:
        """Generate initial solution using greedy approach."""
        solution = self.create_empty_solution()

        # Try to assign courses greedily
        course_order = list(range(1, self.problem.N + 1))
        random.shuffle(course_order)

        for course_id in course_order:
            course_idx = course_id - 1
            duration = self.problem.course_durations[course_idx]
            max_participants = self.problem.course_max_participants[course_idx]

            # Use 80-100% of max participants
            participants = random.randint(
                max(1, int(max_participants * 0.8)),
                max_participants
            )

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

            # Try random assignments
            for attempt in range(30):
                room_id = random.choice(compatible_rooms)
                instructor_id = random.choice(compatible_instructors)
                start_time = random.randint(1, max(1, self.problem.T - duration))

                # Check instructor availability
                instructor_idx = instructor_id - 1
                available = True
                for t_offset in range(duration):
                    t = start_time + t_offset - 1
                    if t >= self.problem.T or self.problem.A_matrix[instructor_idx, t] == 0:
                        available = False
                        break

                if not available:
                    continue

                if solution.assign_course(course_id, room_id, instructor_id,
                                          start_time, duration, participants):
                    break

        return solution

    def improve_solution(self, solution: Solution) -> Solution:
        """
        Single improvement step (not used in standard SA).

        Args:
            solution: Solution to improve

        Returns:
            Neighbor solution
        """
        return self._generate_neighbor(solution)

    def _evaluate_solution(self, solution: Solution) -> float:
        """Evaluate solution fitness with penalties for infeasibility."""
        x, y, u = solution.to_arrays()

        # Calculate base objective
        objective = self.problem.calculate_objective(x, y, u)

        # Add penalties for constraint violations
        is_feasible, violations = self.problem.validate_solution(x, y, u)

        if not is_feasible:
            # Heavy penalty for infeasible solutions
            penalty = len(violations) * 500
            fitness = objective - penalty
        else:
            fitness = objective

        # Update evaluation count
        self.stats.evaluations += 1

        return fitness

    def _generate_neighbor(self, solution: Solution) -> Solution:
        """Generate neighboring solution based on selected method."""
        if self.neighborhood_method == NeighborhoodMethod.SWAP_COURSES:
            return self._swap_neighbor(solution)
        elif self.neighborhood_method == NeighborhoodMethod.RELOCATE_COURSE:
            return self._relocate_neighbor(solution)
        elif self.neighborhood_method == NeighborhoodMethod.TIME_SHIFT:
            return self._time_shift_neighbor(solution)
        elif self.neighborhood_method == NeighborhoodMethod.INSTRUCTOR_CHANGE:
            return self._instructor_change_neighbor(solution)
        elif self.neighborhood_method == NeighborhoodMethod.ADD_REMOVE:
            return self._add_remove_neighbor(solution)
        else:  # MIXED
            return self._mixed_neighbor(solution)

    def _swap_neighbor(self, solution: Solution) -> Solution:
        """Create neighbor by swapping two course assignments."""
        neighbor = solution.copy()
        assignments = neighbor.get_all_assignments()

        if len(assignments) >= 2:
            # Pick two random courses to swap
            course1, course2 = random.sample(assignments, 2)
            neighbor.swap_courses(course1['course_id'], course2['course_id'])

        return neighbor

    def _relocate_neighbor(self, solution: Solution) -> Solution:
        """Create neighbor by relocating a course to different room/time."""
        neighbor = solution.copy()
        assignments = neighbor.get_all_assignments()

        if assignments:
            # Pick random course to relocate
            assignment = random.choice(assignments)
            course_id = assignment['course_id']
            course_idx = course_id - 1

            # Remove current assignment
            neighbor.remove_course_assignment(course_id)

            # Find new location
            duration = self.problem.course_durations[course_idx]
            participants = assignment['participants']

            # Try multiple relocation attempts
            for attempt in range(20):
                # Find compatible room
                compatible_rooms = []
                for room_id in range(1, self.problem.M + 1):
                    room_idx = room_id - 1
                    if (self.problem.R_matrix[course_idx, room_idx] == 1 and
                            self.problem.room_capacities[room_idx] >= participants):
                        compatible_rooms.append(room_id)

                # Find compatible instructor
                compatible_instructors = []
                for instructor_id in range(1, self.problem.K + 1):
                    instructor_idx = instructor_id - 1
                    if self.problem.I_matrix[instructor_idx, course_idx] == 1:
                        compatible_instructors.append(instructor_id)

                if compatible_rooms and compatible_instructors:
                    room_id = random.choice(compatible_rooms)
                    instructor_id = random.choice(compatible_instructors)
                    start_time = random.randint(1, max(1, self.problem.T - duration))

                    if neighbor.assign_course(course_id, room_id, instructor_id,
                                              start_time, duration, participants):
                        break
            else:
                # If relocation failed, restore original assignment
                neighbor.assign_course(
                    course_id, assignment['room_id'], assignment['instructor_id'],
                    assignment['start_time'], assignment['duration'], participants
                )

        return neighbor

    def _time_shift_neighbor(self, solution: Solution) -> Solution:
        """Create neighbor by shifting a course in time."""
        neighbor = solution.copy()
        assignments = neighbor.get_all_assignments()

        if assignments:
            assignment = random.choice(assignments)
            course_id = assignment['course_id']
            duration = assignment['duration']

            # Remove current assignment
            neighbor.remove_course_assignment(course_id)

            # Try new time slots
            for attempt in range(15):
                new_start_time = random.randint(1, max(1, self.problem.T - duration))

                if neighbor.assign_course(
                        course_id, assignment['room_id'], assignment['instructor_id'],
                        new_start_time, duration, assignment['participants']
                ):
                    break
            else:
                # Restore original if shift failed
                neighbor.assign_course(
                    course_id, assignment['room_id'], assignment['instructor_id'],
                    assignment['start_time'], duration, assignment['participants']
                )

        return neighbor

    def _instructor_change_neighbor(self, solution: Solution) -> Solution:
        """Create neighbor by changing instructor for a course."""
        neighbor = solution.copy()
        assignments = neighbor.get_all_assignments()

        if assignments:
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
                # Remove current assignment
                neighbor.remove_course_assignment(course_id)

                # Try new instructor
                new_instructor_id = random.choice(compatible_instructors)

                success = neighbor.assign_course(
                    course_id, assignment['room_id'], new_instructor_id,
                    assignment['start_time'], assignment['duration'],
                    assignment['participants']
                )

                if not success:
                    # Restore original if change failed
                    neighbor.assign_course(
                        course_id, assignment['room_id'], assignment['instructor_id'],
                        assignment['start_time'], assignment['duration'],
                        assignment['participants']
                    )

        return neighbor

    def _add_remove_neighbor(self, solution: Solution) -> Solution:
        """Create neighbor by adding or removing a course."""
        neighbor = solution.copy()

        if random.random() < 0.5:
            # Try to add unscheduled course
            unscheduled = neighbor.get_unscheduled_courses()
            if unscheduled:
                course_id = random.choice(list(unscheduled))
                course_idx = course_id - 1
                duration = self.problem.course_durations[course_idx]
                participants = self.problem.course_max_participants[course_idx]

                # Try to assign
                for attempt in range(15):
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

                    if compatible_rooms and compatible_instructors:
                        room_id = random.choice(compatible_rooms)
                        instructor_id = random.choice(compatible_instructors)
                        start_time = random.randint(1, max(1, self.problem.T - duration))

                        if neighbor.assign_course(course_id, room_id, instructor_id,
                                                  start_time, duration, participants):
                            break
        else:
            # Remove random scheduled course
            assignments = neighbor.get_all_assignments()
            if assignments:
                assignment = random.choice(assignments)
                neighbor.remove_course_assignment(assignment['course_id'])

        return neighbor

    def _mixed_neighbor(self, solution: Solution) -> Solution:
        """Create neighbor using random neighborhood operation."""
        operations = [
            self._swap_neighbor,
            self._relocate_neighbor,
            self._time_shift_neighbor,
            self._instructor_change_neighbor
        ]

        # Add/remove with lower probability
        if random.random() < 0.2:
            operations.append(self._add_remove_neighbor)

        operation = random.choice(operations)
        return operation(solution)

    def _accept_move(self, delta_E: float, temperature: float) -> bool:
        """Decide whether to accept a move based on temperature."""
        if delta_E > 0:
            # Better solution - always accept
            return True
        elif temperature <= 0:
            # No temperature - only accept improvements
            return False
        else:
            # Worse solution - accept with probability
            probability = math.exp(delta_E / temperature)
            return random.random() < probability

    def _cool_temperature(self, current_temp: float) -> float:
        """Apply cooling schedule to reduce temperature."""
        if self.cooling_schedule == CoolingSchedule.EXPONENTIAL:
            return current_temp * self.cooling_rate

        elif self.cooling_schedule == CoolingSchedule.LINEAR:
            decrease = (self.initial_temperature - self.final_temperature) / self.max_iterations
            return max(self.final_temperature, current_temp - decrease)

        elif self.cooling_schedule == CoolingSchedule.LOGARITHMIC:
            return self.initial_temperature / math.log(1 + self.iteration)

        else:  # ADAPTIVE handled separately
            return current_temp * self.cooling_rate

    def _adjust_temperature_adaptive(self, acceptance_rate: float) -> None:
        """Adjust temperature based on acceptance rate."""
        if acceptance_rate > self.acceptance_rate_target:
            # Too many acceptances - cool down faster
            self.current_temperature *= (self.cooling_rate / self.temperature_adjustment_factor)
        elif acceptance_rate < self.acceptance_rate_target * 0.5:
            # Too few acceptances - cool down slower
            self.current_temperature *= (self.cooling_rate * self.temperature_adjustment_factor)
        else:
            # Normal cooling
            self.current_temperature *= self.cooling_rate

        # Ensure temperature doesn't go below minimum
        self.current_temperature = max(self.final_temperature, self.current_temperature)

    def _restart(self) -> None:
        """Restart with new solution and reset temperature."""
        if self.verbose:
            print(f"  Restarting from stagnation (restart #{self.restarts_used + 1})")

        # Generate new starting solution
        self.current_solution = self.generate_initial_solution()
        self.current_fitness = self._evaluate_solution(self.current_solution)

        # Reset temperature
        self.current_temperature = self.initial_temperature

        # Reset counters
        self.stagnation_counter = 0
        self.restarts_used += 1


def create_sa_variants() -> List[Dict]:
    """
    Create different simulated annealing variants with various parameter combinations.

    Returns:
        List of parameter dictionaries for different SA variants
    """
    variants = []

    # Classic SA with exponential cooling
    variants.append({
        'name': 'SA_Classic_Exponential',
        'initial_temperature': 1000.0,
        'final_temperature': 0.1,
        'cooling_rate': 0.95,
        'cooling_schedule': 'exponential',
        'neighborhood_method': 'mixed',
        'iterations_per_temp': 50,
        'max_iterations': 5000
    })

    # Fast SA with aggressive cooling
    variants.append({
        'name': 'SA_Fast_Aggressive',
        'initial_temperature': 500.0,
        'final_temperature': 0.5,
        'cooling_rate': 0.9,
        'cooling_schedule': 'exponential',
        'neighborhood_method': 'swap_courses',
        'iterations_per_temp': 25,
        'max_iterations': 3000
    })

    # Adaptive SA
    variants.append({
        'name': 'SA_Adaptive',
        'initial_temperature': 2000.0,
        'final_temperature': 0.01,
        'cooling_rate': 0.98,
        'cooling_schedule': 'adaptive',
        'neighborhood_method': 'mixed',
        'iterations_per_temp': 75,
        'adaptive_cooling': True,
        'acceptance_rate_target': 0.4,
        'max_iterations': 8000
    })

    # SA with restarts
    variants.append({
        'name': 'SA_WithRestarts',
        'initial_temperature': 800.0,
        'final_temperature': 0.1,
        'cooling_rate': 0.92,
        'cooling_schedule': 'exponential',
        'neighborhood_method': 'relocate_course',
        'iterations_per_temp': 40,
        'restart_threshold': 500,
        'max_restarts': 2,
        'max_iterations': 6000
    })

    # High temperature SA for exploration
    variants.append({
        'name': 'SA_HighTemp_Exploration',
        'initial_temperature': 5000.0,
        'final_temperature': 1.0,
        'cooling_rate': 0.99,
        'cooling_schedule': 'logarithmic',
        'neighborhood_method': 'mixed',
        'iterations_per_temp': 100,
        'max_iterations': 10000
    })

    return variants


if __name__ == "__main__":
    # Test simulated annealing algorithm
    import sys
    import os

    # Import problem
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from utils.data_loader import load_instance

    try:
        # Load test instance
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        instance_path = os.path.join(project_root, "data", "instances", "small", "small_01")
        loader = load_instance(instance_path)
        problem = CourseSchedulingProblem(loader)

        print("Testing Simulated Annealing variants...")

        # Test different variants
        variants = create_sa_variants()

        for variant in variants[:2]:  # Test first 2 variants
            print(f"\n{'=' * 60}")
            print(f"Testing: {variant['name']}")
            print(f"{'=' * 60}")

            # Create and run algorithm
            sa = SimulatedAnnealingAlgorithm(problem, verbose=True, **variant)
            solution = sa.run()

            # Print results
            sa.print_statistics()

            # Validate solution
            x, y, u = solution.to_arrays()
            is_feasible, violations = problem.validate_solution(x, y, u)

            print(f"\nSolution validation:")
            print(f"  Feasible: {is_feasible}")
            if violations:
                print(f"  Violations: {len(violations)}")
                for violation in violations[:3]:  # Show first 3
                    print(f"    - {violation}")

            # Print cooling info
            if sa.temperature_history:
                print(f"\nCooling statistics:")
                print(f"  Initial temperature: {sa.temperature_history[0]:.2f}")
                print(f"  Final temperature: {sa.temperature_history[-1]:.4f}")
                if sa.acceptance_history:
                    avg_acceptance = np.mean(sa.acceptance_history)
                    print(f"  Average acceptance rate: {avg_acceptance:.2%}")

        print(f"\n✓ Simulated annealing testing completed!")

    except Exception as e:
        print(f"Error testing simulated annealing: {e}")
        import traceback

        traceback.print_exc()