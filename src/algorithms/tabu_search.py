import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
from collections import deque
from dataclasses import dataclass

from models.problem import CourseSchedulingProblem
from models.solution import Solution
from base import ImprovementAlgorithm


class NeighborhoodStrategy(Enum):
    """Strategies for exploring neighborhood."""
    BEST_IMPROVEMENT = "best_improvement"
    FIRST_IMPROVEMENT = "first_improvement"
    CANDIDATE_LIST = "candidate_list"
    RANDOM_SAMPLING = "random_sampling"


class DiversificationMethod(Enum):
    """Methods for diversification when stagnating."""
    RESTART = "restart"
    BIG_PERTURBATION = "big_perturbation"
    FREQUENCY_BASED = "frequency_based"
    ELITE_SOLUTIONS = "elite_solutions"


@dataclass
class TabuMove:
    """Represents a move in the tabu list."""
    move_type: str
    course_id: int
    from_room: Optional[int] = None
    to_room: Optional[int] = None
    from_instructor: Optional[int] = None
    to_instructor: Optional[int] = None
    from_time: Optional[int] = None
    to_time: Optional[int] = None

    def get_reverse_move(self) -> 'TabuMove':
        """Get the reverse of this move."""
        return TabuMove(
            move_type=self.move_type,
            course_id=self.course_id,
            from_room=self.to_room,
            to_room=self.from_room,
            from_instructor=self.to_instructor,
            to_instructor=self.from_instructor,
            from_time=self.to_time,
            to_time=self.from_time
        )

    def __eq__(self, other) -> bool:
        """Check if two moves are equivalent."""
        if not isinstance(other, TabuMove):
            return False
        return (
                self.move_type == other.move_type and
                self.course_id == other.course_id and
                self.from_room == other.from_room and
                self.to_room == other.to_room and
                self.from_instructor == other.from_instructor and
                self.to_instructor == other.to_instructor and
                self.from_time == other.from_time and
                self.to_time == other.to_time
        )

    def __hash__(self) -> int:
        """Make move hashable for use in sets."""
        return hash((
            self.move_type, self.course_id, self.from_room, self.to_room,
            self.from_instructor, self.to_instructor, self.from_time, self.to_time
        ))


@dataclass
class MoveCandidate:
    """Represents a candidate move with its evaluation."""
    move: TabuMove
    new_solution: Solution
    fitness: float
    is_tabu: bool
    is_aspiration: bool


class TabuSearchAlgorithm(ImprovementAlgorithm):
    """
    Tabu Search algorithm for course scheduling optimization.

    Uses adaptive memory structures and strategic oscillation
    to escape local optima and explore solution space efficiently.
    """

    def __init__(self, problem: CourseSchedulingProblem, **kwargs):
        """
        Initialize tabu search algorithm.

        Args:
            problem: Problem instance
            **kwargs: Algorithm parameters
        """
        # Extract name from kwargs or use default
        algorithm_name = kwargs.pop('name', 'TabuSearchAlgorithm')
        super().__init__(problem, name=algorithm_name, **kwargs)

        # Tabu list parameters
        self.tabu_tenure = kwargs.get('tabu_tenure', 10)
        self.dynamic_tenure = kwargs.get('dynamic_tenure', False)
        self.min_tenure = kwargs.get('min_tenure', 5)
        self.max_tenure = kwargs.get('max_tenure', 20)

        # Search strategy
        self.neighborhood_strategy = NeighborhoodStrategy(
            kwargs.get('neighborhood_strategy', 'best_improvement')
        )
        self.candidate_list_size = kwargs.get('candidate_list_size', 50)

        # Aspiration criteria
        self.aspiration_criteria = kwargs.get('aspiration_criteria', True)
        self.aspiration_threshold = kwargs.get('aspiration_threshold', 0.0)

        # Intensification parameters
        self.intensification_enabled = kwargs.get('intensification_enabled', True)
        self.intensification_threshold = kwargs.get('intensification_threshold', 100)
        self.intensification_iterations = kwargs.get('intensification_iterations', 20)

        # Diversification parameters
        self.diversification_enabled = kwargs.get('diversification_enabled', True)
        self.diversification_threshold = kwargs.get('diversification_threshold', 200)
        self.diversification_method = DiversificationMethod(
            kwargs.get('diversification_method', 'big_perturbation')
        )

        # Elite solutions
        self.elite_size = kwargs.get('elite_size', 5)
        self.elite_solutions: List[Tuple[Solution, float]] = []

        # Random seed
        self.random_seed = kwargs.get('random_seed', None)
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        # Tabu structures
        self.tabu_list: deque = deque(maxlen=self.tabu_tenure)
        self.move_frequency: Dict[str, int] = {}

        # State tracking
        self.current_solution: Optional[Solution] = None
        self.current_fitness = float('-inf')
        self.iteration = 0
        self.stagnation_counter = 0
        self.last_improvement_iteration = 0

        # Statistics
        self.moves_evaluated = 0
        self.tabu_violations = 0
        self.aspirations_used = 0
        self.diversifications_performed = 0
        self.intensifications_performed = 0

    def solve(self) -> Solution:
        """
        Main solving method implementing tabu search.

        Returns:
            Best solution found
        """
        if self.verbose:
            print(f"Starting tabu search:")
            print(f"  Tabu tenure: {self.tabu_tenure}")
            print(f"  Neighborhood strategy: {self.neighborhood_strategy.value}")
            print(f"  Aspiration criteria: {self.aspiration_criteria}")
            print(f"  Intensification: {self.intensification_enabled}")
            print(f"  Diversification: {self.diversification_enabled}")

        # Generate initial solution
        self.current_solution = self.generate_initial_solution()
        self.current_fitness = self._evaluate_solution(self.current_solution)
        self.stats.initial_objective = self.current_fitness

        # Initialize best solution
        self.best_solution = self.current_solution.copy()
        best_fitness = self.current_fitness
        self._add_to_elite(self.current_solution, self.current_fitness)

        # Main tabu search loop
        while self.iteration < self.max_iterations:

            # Generate and evaluate all candidate moves
            candidates = self._generate_candidates()

            if not candidates:
                if self.verbose:
                    print(f"  No valid candidates at iteration {self.iteration}")
                break

            # Select best non-tabu move (or aspiration move)
            best_candidate = self._select_best_candidate(candidates)

            if best_candidate is None:
                if self.verbose:
                    print(f"  No acceptable move at iteration {self.iteration}")
                break

            # Execute move
            self._execute_move(best_candidate)

            # Update best solution
            if best_candidate.fitness > best_fitness:
                self.best_solution = best_candidate.new_solution.copy()
                best_fitness = best_candidate.fitness
                self.stats.improvements += 1
                self.last_improvement_iteration = self.iteration
                self.stagnation_counter = 0

                if self.verbose:
                    print(f"    New best: {best_fitness:.2f}")
            else:
                self.stagnation_counter += 1

            # Add to elite solutions
            self._add_to_elite(best_candidate.new_solution, best_candidate.fitness)

            # Update tabu tenure dynamically
            if self.dynamic_tenure:
                self._update_tabu_tenure()

            # Check for intensification
            if (self.intensification_enabled and
                    self.stagnation_counter >= self.intensification_threshold):
                self._intensify()

            # Check for diversification
            if (self.diversification_enabled and
                    self.stagnation_counter >= self.diversification_threshold):
                self._diversify()

            # Log progress
            self.log_progress(
                self.iteration + 1,
                best_fitness,
                f"(current: {best_candidate.fitness:.2f}, tabu_size: {len(self.tabu_list)})"
            )

            self.iteration += 1

            # Check termination
            if self.should_terminate(self.iteration):
                break

        if self.verbose:
            print(f"\nTabu search completed:")
            print(f"  Total iterations: {self.iteration}")
            print(f"  Moves evaluated: {self.moves_evaluated}")
            print(f"  Tabu violations: {self.tabu_violations}")
            print(f"  Aspirations used: {self.aspirations_used}")
            print(f"  Diversifications: {self.diversifications_performed}")
            print(f"  Intensifications: {self.intensifications_performed}")
            self.best_solution.print_solution_summary()

        return self.best_solution

    def generate_initial_solution(self) -> Solution:
        """Generate initial solution using greedy approach."""
        solution = self.create_empty_solution()

        # Try to assign courses using simple greedy
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
            for attempt in range(25):
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
        """Single improvement step (not used in standard TS)."""
        candidates = self._generate_candidates_for_solution(solution)
        if candidates:
            best_candidate = max(candidates, key=lambda c: c.fitness)
            return best_candidate.new_solution
        return solution

    def _evaluate_solution(self, solution: Solution) -> float:
        """Evaluate solution fitness with penalties."""
        x, y, u = solution.to_arrays()
        objective = self.problem.calculate_objective(x, y, u)

        # Add penalties for infeasibility
        is_feasible, violations = self.problem.validate_solution(x, y, u)
        if not is_feasible:
            penalty = len(violations) * 300
            fitness = objective - penalty
        else:
            fitness = objective

        self.stats.evaluations += 1
        return fitness

    def _generate_candidates(self) -> List[MoveCandidate]:
        """Generate all candidate moves from current solution."""
        return self._generate_candidates_for_solution(self.current_solution)

    def _generate_candidates_for_solution(self, solution: Solution) -> List[MoveCandidate]:
        """Generate candidate moves for given solution."""
        candidates = []
        assignments = solution.get_all_assignments()

        # Swap moves
        for i, assignment1 in enumerate(assignments):
            for j, assignment2 in enumerate(assignments[i + 1:], i + 1):
                move = TabuMove(
                    move_type="swap",
                    course_id=assignment1['course_id'],
                    from_room=assignment1['room_id'],
                    to_room=assignment2['room_id'],
                    from_instructor=assignment1['instructor_id'],
                    to_instructor=assignment2['instructor_id'],
                    from_time=assignment1['start_time'],
                    to_time=assignment2['start_time']
                )

                candidate = self._evaluate_move(solution, move)
                if candidate:
                    candidates.append(candidate)

        # Relocate moves (change room)
        for assignment in assignments:
            course_id = assignment['course_id']
            course_idx = course_id - 1
            current_room = assignment['room_id']
            participants = assignment['participants']

            for room_id in range(1, self.problem.M + 1):
                if room_id == current_room:
                    continue

                room_idx = room_id - 1
                if (self.problem.R_matrix[course_idx, room_idx] == 1 and
                        self.problem.room_capacities[room_idx] >= participants):

                    move = TabuMove(
                        move_type="relocate_room",
                        course_id=course_id,
                        from_room=current_room,
                        to_room=room_id
                    )

                    candidate = self._evaluate_move(solution, move)
                    if candidate:
                        candidates.append(candidate)

        # Change instructor moves
        for assignment in assignments:
            course_id = assignment['course_id']
            course_idx = course_id - 1
            current_instructor = assignment['instructor_id']

            for instructor_id in range(1, self.problem.K + 1):
                if instructor_id == current_instructor:
                    continue

                instructor_idx = instructor_id - 1
                if self.problem.I_matrix[instructor_idx, course_idx] == 1:

                    move = TabuMove(
                        move_type="change_instructor",
                        course_id=course_id,
                        from_instructor=current_instructor,
                        to_instructor=instructor_id
                    )

                    candidate = self._evaluate_move(solution, move)
                    if candidate:
                        candidates.append(candidate)

        # Time shift moves
        for assignment in assignments:
            course_id = assignment['course_id']
            current_time = assignment['start_time']
            duration = assignment['duration']

            for new_time in range(1, self.problem.T - duration + 2):
                if new_time == current_time:
                    continue

                move = TabuMove(
                    move_type="time_shift",
                    course_id=course_id,
                    from_time=current_time,
                    to_time=new_time
                )

                candidate = self._evaluate_move(solution, move)
                if candidate:
                    candidates.append(candidate)

        # Add/remove moves
        unscheduled = solution.get_unscheduled_courses()
        if unscheduled:
            # Try to add unscheduled courses
            for course_id in list(unscheduled)[:5]:  # Limit to avoid too many candidates
                move = TabuMove(
                    move_type="add_course",
                    course_id=course_id
                )

                candidate = self._evaluate_move(solution, move)
                if candidate:
                    candidates.append(candidate)

        if assignments:
            # Try to remove scheduled courses
            for assignment in assignments[:3]:  # Limit removals
                move = TabuMove(
                    move_type="remove_course",
                    course_id=assignment['course_id']
                )

                candidate = self._evaluate_move(solution, move)
                if candidate:
                    candidates.append(candidate)

        self.moves_evaluated += len(candidates)

        # Apply neighborhood strategy
        if self.neighborhood_strategy == NeighborhoodStrategy.CANDIDATE_LIST:
            # Keep only best candidates
            candidates.sort(key=lambda c: c.fitness, reverse=True)
            candidates = candidates[:self.candidate_list_size]
        elif self.neighborhood_strategy == NeighborhoodStrategy.RANDOM_SAMPLING:
            # Random sample
            if len(candidates) > self.candidate_list_size:
                candidates = random.sample(candidates, self.candidate_list_size)

        return candidates

    def _evaluate_move(self, solution: Solution, move: TabuMove) -> Optional[MoveCandidate]:
        """Evaluate a specific move and return candidate."""
        try:
            new_solution = self._apply_move(solution, move)
            if new_solution is None:
                return None

            fitness = self._evaluate_solution(new_solution)
            is_tabu = self._is_move_tabu(move)
            is_aspiration = self._check_aspiration(move, fitness)

            return MoveCandidate(
                move=move,
                new_solution=new_solution,
                fitness=fitness,
                is_tabu=is_tabu,
                is_aspiration=is_aspiration
            )

        except Exception:
            return None

    def _apply_move(self, solution: Solution, move: TabuMove) -> Optional[Solution]:
        """Apply move to solution and return new solution."""
        new_solution = solution.copy()

        try:
            if move.move_type == "swap":
                success = new_solution.swap_courses(move.course_id,
                                                    self._find_course_at_position(solution, move.to_room, move.to_time))
                if not success:
                    return None

            elif move.move_type == "relocate_room":
                assignment = new_solution.get_course_assignment(move.course_id)
                if assignment:
                    new_solution.remove_course_assignment(move.course_id)
                    success = new_solution.assign_course(
                        move.course_id, move.to_room, assignment['instructor_id'],
                        assignment['start_time'], assignment['duration'], assignment['participants']
                    )
                    if not success:
                        return None

            elif move.move_type == "change_instructor":
                assignment = new_solution.get_course_assignment(move.course_id)
                if assignment:
                    new_solution.remove_course_assignment(move.course_id)
                    success = new_solution.assign_course(
                        move.course_id, assignment['room_id'], move.to_instructor,
                        assignment['start_time'], assignment['duration'], assignment['participants']
                    )
                    if not success:
                        return None

            elif move.move_type == "time_shift":
                assignment = new_solution.get_course_assignment(move.course_id)
                if assignment:
                    new_solution.remove_course_assignment(move.course_id)
                    success = new_solution.assign_course(
                        move.course_id, assignment['room_id'], assignment['instructor_id'],
                        move.to_time, assignment['duration'], assignment['participants']
                    )
                    if not success:
                        return None

            elif move.move_type == "add_course":
                # Try to add unscheduled course
                course_idx = move.course_id - 1
                duration = self.problem.course_durations[course_idx]
                participants = self.problem.course_max_participants[course_idx]

                # Find available resources
                for attempt in range(10):
                    room_id = random.randint(1, self.problem.M)
                    instructor_id = random.randint(1, self.problem.K)
                    start_time = random.randint(1, max(1, self.problem.T - duration))

                    if new_solution.assign_course(move.course_id, room_id, instructor_id,
                                                  start_time, duration, participants):
                        break
                else:
                    return None

            elif move.move_type == "remove_course":
                new_solution.remove_course_assignment(move.course_id)

            return new_solution

        except Exception:
            return None

    def _find_course_at_position(self, solution: Solution, room_id: int, time_slot: int) -> Optional[int]:
        """Find course at specific room and time."""
        for assignment in solution.get_all_assignments():
            if (assignment['room_id'] == room_id and
                    assignment['start_time'] <= time_slot <= assignment['end_time']):
                return assignment['course_id']
        return None

    def _is_move_tabu(self, move: TabuMove) -> bool:
        """Check if move is in tabu list."""
        reverse_move = move.get_reverse_move()
        return reverse_move in self.tabu_list

    def _check_aspiration(self, move: TabuMove, fitness: float) -> bool:
        """Check if move satisfies aspiration criteria."""
        if not self.aspiration_criteria:
            return False

        # Aspiration: accept tabu move if it's better than current best
        current_best = self.best_solution
        if current_best:
            best_fitness = self._evaluate_solution(current_best)
            return fitness > best_fitness + self.aspiration_threshold

        return False

    def _select_best_candidate(self, candidates: List[MoveCandidate]) -> Optional[MoveCandidate]:
        """Select best candidate according to tabu rules."""
        if not candidates:
            return None

        # Separate non-tabu and aspiration candidates
        non_tabu = [c for c in candidates if not c.is_tabu]
        aspiration = [c for c in candidates if c.is_tabu and c.is_aspiration]

        # Strategy selection
        if self.neighborhood_strategy == NeighborhoodStrategy.FIRST_IMPROVEMENT:
            # Return first improving move
            for candidate in candidates:
                if candidate.fitness > self.current_fitness and not candidate.is_tabu:
                    return candidate
            # If no improvement, take best non-tabu
            if non_tabu:
                return max(non_tabu, key=lambda c: c.fitness)

        # For other strategies, prefer non-tabu moves
        if non_tabu:
            best_candidate = max(non_tabu, key=lambda c: c.fitness)
        elif aspiration:
            best_candidate = max(aspiration, key=lambda c: c.fitness)
            self.aspirations_used += 1
        else:
            # All moves are tabu and no aspiration
            self.tabu_violations += 1
            return None

        return best_candidate

    def _execute_move(self, candidate: MoveCandidate) -> None:
        """Execute selected move and update state."""
        # Update current solution
        self.current_solution = candidate.new_solution
        self.current_fitness = candidate.fitness

        # Add move to tabu list
        self.tabu_list.append(candidate.move)

        # Update move frequency
        move_key = f"{candidate.move.move_type}_{candidate.move.course_id}"
        self.move_frequency[move_key] = self.move_frequency.get(move_key, 0) + 1

    def _update_tabu_tenure(self) -> None:
        """Dynamically adjust tabu tenure."""
        if self.stagnation_counter > 50:
            # Increase tenure when stagnating
            new_tenure = min(self.max_tenure, self.tabu_tenure + 1)
        elif self.stagnation_counter < 10:
            # Decrease tenure when improving
            new_tenure = max(self.min_tenure, self.tabu_tenure - 1)
        else:
            new_tenure = self.tabu_tenure

        if new_tenure != self.tabu_tenure:
            self.tabu_tenure = new_tenure
            self.tabu_list = deque(list(self.tabu_list), maxlen=new_tenure)

    def _add_to_elite(self, solution: Solution, fitness: float) -> None:
        """Add solution to elite set."""
        self.elite_solutions.append((solution.copy(), fitness))
        self.elite_solutions.sort(key=lambda x: x[1], reverse=True)
        self.elite_solutions = self.elite_solutions[:self.elite_size]

    def _intensify(self) -> None:
        """Perform intensification around best solutions."""
        if not self.elite_solutions:
            return

        if self.verbose:
            print(f"  Intensifying around elite solutions...")

        # Start from best elite solution
        best_elite_solution, _ = self.elite_solutions[0]
        self.current_solution = best_elite_solution.copy()
        self.current_fitness = self._evaluate_solution(self.current_solution)

        # Clear tabu list for fresh exploration
        self.tabu_list.clear()

        # Reset counters
        self.stagnation_counter = 0
        self.intensifications_performed += 1

    def _diversify(self) -> None:
        """Perform diversification to explore new regions."""
        if self.verbose:
            print(f"  Diversifying search...")

        if self.diversification_method == DiversificationMethod.RESTART:
            # Complete restart
            self.current_solution = self.generate_initial_solution()
            self.current_fitness = self._evaluate_solution(self.current_solution)

        elif self.diversification_method == DiversificationMethod.BIG_PERTURBATION:
            # Apply multiple random moves
            for _ in range(5):
                candidates = self._generate_candidates()
                if candidates:
                    # Pick random candidate (not necessarily best)
                    candidate = random.choice(candidates)
                    self.current_solution = candidate.new_solution
                    self.current_fitness = candidate.fitness

        elif self.diversification_method == DiversificationMethod.FREQUENCY_BASED:
            # Avoid frequently used moves
            self._penalize_frequent_moves()

        elif self.diversification_method == DiversificationMethod.ELITE_SOLUTIONS:
            # Start from random elite solution
            if self.elite_solutions:
                elite_solution, _ = random.choice(self.elite_solutions)
                self.current_solution = elite_solution.copy()
                self.current_fitness = self._evaluate_solution(self.current_solution)

        # Clear tabu list and reset counters
        self.tabu_list.clear()
        self.move_frequency.clear()
        self.stagnation_counter = 0
        self.diversifications_performed += 1

    def _penalize_frequent_moves(self) -> None:
        """Modify solution to avoid frequently used moves."""
        # Find most frequent moves
        if not self.move_frequency:
            return

        frequent_moves = sorted(self.move_frequency.items(),
                                key=lambda x: x[1], reverse=True)

        # Apply opposite moves to diversify
        for move_key, frequency in frequent_moves[:3]:
            move_parts = move_key.split('_')
            if len(move_parts) >= 2:
                move_type = move_parts[0]
                course_id = int(move_parts[1])

                # Apply diversification move
                assignment = self.current_solution.get_course_assignment(course_id)
                if assignment and move_type in ['relocate', 'change']:
                    # Try different assignment
                    self.current_solution.remove_course_assignment(course_id)
                    # Try random reassignment
                    for attempt in range(5):
                        room_id = random.randint(1, self.problem.M)
                        instructor_id = random.randint(1, self.problem.K)
                        if self.current_solution.assign_course(
                                course_id, room_id, instructor_id,
                                assignment['start_time'], assignment['duration'],
                                assignment['participants']
                        ):
                            break

        # Update fitness after diversification changes
        self.current_fitness = self._evaluate_solution(self.current_solution)

def create_tabu_variants() -> List[Dict]:
    """
    Create different tabu search variants with various parameter combinations.

    Returns:
        List of parameter dictionaries for different TS variants
    """
    variants = []

    # Classic Tabu Search
    variants.append({
        'name': 'TS_Classic',
        'tabu_tenure': 10,
        'neighborhood_strategy': 'best_improvement',
        'aspiration_criteria': True,
        'intensification_enabled': True,
        'diversification_enabled': True,
        'intensification_threshold': 100,
        'diversification_threshold': 200,
        'diversification_method': 'big_perturbation',
        'max_iterations': 1000
    })

    # Adaptive Tabu Search with dynamic tenure
    variants.append({
        'name': 'TS_Adaptive',
        'tabu_tenure': 15,
        'dynamic_tenure': True,
        'min_tenure': 5,
        'max_tenure': 25,
        'neighborhood_strategy': 'candidate_list',
        'candidate_list_size': 50,
        'aspiration_criteria': True,
        'intensification_enabled': True,
        'diversification_enabled': True,
        'intensification_threshold': 75,
        'diversification_threshold': 150,
        'diversification_method': 'frequency_based',
        'max_iterations': 1500
    })

    # Fast Tabu Search
    variants.append({
        'name': 'TS_Fast',
        'tabu_tenure': 7,
        'neighborhood_strategy': 'first_improvement',
        'aspiration_criteria': True,
        'intensification_enabled': False,
        'diversification_enabled': True,
        'diversification_threshold': 100,
        'diversification_method': 'restart',
        'max_iterations': 800
    })

    # Elite-based Tabu Search
    variants.append({
        'name': 'TS_Elite',
        'tabu_tenure': 12,
        'neighborhood_strategy': 'best_improvement',
        'aspiration_criteria': True,
        'elite_size': 10,
        'intensification_enabled': True,
        'diversification_enabled': True,
        'intensification_threshold': 80,
        'diversification_threshold': 180,
        'diversification_method': 'elite_solutions',
        'max_iterations': 1200
    })

    # Aggressive exploration Tabu Search
    variants.append({
        'name': 'TS_Aggressive',
        'tabu_tenure': 20,
        'neighborhood_strategy': 'random_sampling',
        'candidate_list_size': 30,
        'aspiration_criteria': True,
        'aspiration_threshold': 10.0,
        'intensification_enabled': True,
        'diversification_enabled': True,
        'intensification_threshold': 50,
        'diversification_threshold': 120,
        'diversification_method': 'big_perturbation',
        'max_iterations': 2000
    })

    return variants

if __name__ == "__main__":
    # Test tabu search algorithm
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

        print("Testing Tabu Search variants...")

        # Test different variants
        variants = create_tabu_variants()

        for variant in variants[:2]:  # Test first 2 variants
            print(f"\n{'=' * 60}")
            print(f"Testing: {variant['name']}")
            print(f"{'=' * 60}")

            # Create and run algorithm
            ts = TabuSearchAlgorithm(problem, verbose=True, **variant)
            solution = ts.run()

            # Print results
            ts.print_statistics()

            # Validate solution
            x, y, u = solution.to_arrays()
            is_feasible, violations = problem.validate_solution(x, y, u)

            print(f"\nSolution validation:")
            print(f"  Feasible: {is_feasible}")
            if violations:
                print(f"  Violations: {len(violations)}")
                for violation in violations[:3]:  # Show first 3
                    print(f"    - {violation}")

            # Print tabu-specific stats
            print(f"\nTabu Search statistics:")
            print(f"  Final tabu list size: {len(ts.tabu_list)}")
            print(f"  Elite solutions found: {len(ts.elite_solutions)}")
            if ts.elite_solutions:
                best_elite_fitness = ts.elite_solutions[0][1]
                print(f"  Best elite fitness: {best_elite_fitness:.2f}")

        print(f"\n✓ Tabu search testing completed!")

    except Exception as e:
        print(f"Error testing tabu search: {e}")
        import traceback
        traceback.print_exc()