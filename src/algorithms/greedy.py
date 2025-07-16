import numpy as np
from typing import List, Dict, Optional, Tuple
import random
from enum import Enum

from models.problem import CourseSchedulingProblem
from models.solution import Solution
from base import ConstructiveAlgorithm


class CourseSelectionStrategy(Enum):
    """Strategies for selecting which course to assign next."""
    PROFIT_DENSITY = "profit_density"  # Revenue per hour
    HIGHEST_PRICE = "highest_price"  # Course with highest price first
    LONGEST_DURATION = "longest_duration"  # Longest courses first
    SHORTEST_DURATION = "shortest_duration"  # Shortest courses first
    MOST_PARTICIPANTS = "most_participants"  # Highest capacity courses first
    MOST_CONSTRAINED = "most_constrained"  # Fewest available options first
    RANDOM = "random"  # Random order


class ResourceSelectionStrategy(Enum):
    """Strategies for selecting rooms and instructors."""
    CHEAPEST = "cheapest"  # Lowest cost option
    EARLIEST_AVAILABLE = "earliest_available"  # First available time slot
    BEST_FIT_CAPACITY = "best_fit_capacity"  # Room capacity closest to participants
    HIGHEST_AVAILABILITY = "highest_availability"  # Most available resource
    RANDOM = "random"  # Random selection


class ParticipantStrategy(Enum):
    """Strategies for determining number of participants."""
    MAXIMUM = "maximum"  # Use maximum allowed
    PERCENTAGE = "percentage"  # Use percentage of maximum
    FIXED = "fixed"  # Use fixed number
    RANDOM = "random"  # Random within limits


class GreedyAlgorithm(ConstructiveAlgorithm):
    """
    Greedy algorithm for course scheduling.

    Builds solution constructively by assigning courses one by one,
    making locally optimal choices at each step.
    """

    def __init__(self, problem: CourseSchedulingProblem, **kwargs):
        """
        Initialize greedy algorithm.

        Args:
            problem: Problem instance
            **kwargs: Algorithm parameters
        """
        # Extract name from kwargs or use default
        algorithm_name = kwargs.pop('name', 'GreedyAlgorithm')
        super().__init__(problem, name=algorithm_name, **kwargs)

        # Strategy parameters
        self.course_strategy = CourseSelectionStrategy(
            kwargs.get('course_selection_strategy', 'profit_density')
        )
        self.room_strategy = ResourceSelectionStrategy(
            kwargs.get('room_selection_strategy', 'cheapest')
        )
        self.instructor_strategy = ResourceSelectionStrategy(
            kwargs.get('instructor_selection_strategy', 'cheapest')
        )
        self.participant_strategy = ParticipantStrategy(
            kwargs.get('participant_strategy', 'maximum')
        )

        # Participant strategy parameters
        self.participant_percentage = kwargs.get('participant_percentage', 0.8)
        self.fixed_participants = kwargs.get('fixed_participants', 10)

        # Random seed for reproducibility
        self.random_seed = kwargs.get('random_seed', None)
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        # Lookahead parameter (how many time slots to look ahead)
        self.lookahead = kwargs.get('lookahead', 5)

    def solve(self) -> Solution:
        """
        Main solving method implementing greedy construction.

        Returns:
            Constructed solution
        """
        if self.verbose:
            print(f"Starting greedy algorithm with strategies:")
            print(f"  Course selection: {self.course_strategy.value}")
            print(f"  Room selection: {self.room_strategy.value}")
            print(f"  Instructor selection: {self.instructor_strategy.value}")
            print(f"  Participant strategy: {self.participant_strategy.value}")

        # Create empty solution
        solution = self.create_empty_solution()

        # Get ordered list of courses to assign
        course_order = self._get_course_order()

        assigned_courses = 0
        for iteration, course_id in enumerate(course_order, 1):

            # Try to assign this course
            assignment = self.select_assignment(solution, course_id)

            if assignment:
                # Make the assignment
                success = solution.assign_course(
                    course_id=course_id,
                    room_id=assignment['room_id'],
                    instructor_id=assignment['instructor_id'],
                    start_time=assignment['start_time'],
                    duration=assignment['duration'],
                    participants=assignment['participants']
                )

                if success:
                    assigned_courses += 1

                    # Evaluate current solution
                    objective = self.evaluate_solution(solution)

                    # Log progress
                    self.log_progress(
                        iteration,
                        objective,
                        f"(assigned course {course_id}, total: {assigned_courses}/{self.problem.N})"
                    )
                else:
                    if self.verbose:
                        print(f"  Failed to assign course {course_id} (assignment conflict)")
            else:
                if self.verbose:
                    print(f"  No valid assignment found for course {course_id}")

        if self.verbose:
            print(f"\nGreedy construction completed:")
            print(f"  Assigned courses: {assigned_courses}/{self.problem.N}")
            solution.print_solution_summary()

        return solution

    def select_next_course(self, solution: Solution) -> Optional[int]:
        """
        Select next course to assign based on strategy.

        Args:
            solution: Current partial solution

        Returns:
            Course ID to assign next, or None if all assigned
        """
        unscheduled = solution.get_unscheduled_courses()
        if not unscheduled:
            return None

        return self._select_course_by_strategy(list(unscheduled), solution)

    def select_assignment(self, solution: Solution, course_id: int) -> Optional[Dict]:
        """
        Select room, instructor, time and participants for given course.

        Args:
            solution: Current solution
            course_id: Course to assign

        Returns:
            Assignment dictionary or None if no valid assignment
        """
        course_idx = course_id - 1
        duration = self.problem.course_durations[course_idx]

        # Determine number of participants
        participants = self._determine_participants(course_id)

        # Get compatible rooms and instructors
        compatible_rooms = self._get_compatible_rooms(course_id, participants)
        compatible_instructors = self._get_compatible_instructors(course_id)


        if self.verbose:
            print(f"    Course {course_id}: duration={duration}, participants={participants}")
            print(f"    Compatible rooms: {compatible_rooms}")
            print(f"    Compatible instructors: {compatible_instructors}")

        if not compatible_rooms or not compatible_instructors:
            if self.verbose:
                print(f"    ❌ No compatible rooms or instructors for course {course_id}")
            return None

        # Try to find valid assignment
        best_assignment = None
        best_score = float('-inf')

        # Order rooms and instructors by strategy
        ordered_rooms = self._order_rooms_by_strategy(compatible_rooms, course_id, participants)
        ordered_instructors = self._order_instructors_by_strategy(compatible_instructors, course_id)

        # Try combinations of room, instructor, and time
        for room_id in ordered_rooms:
            for instructor_id in ordered_instructors:

                # Find available time slots
                available_times = self._find_available_times(
                    solution, room_id, instructor_id, duration
                )

                if self.verbose:
                    print(f"    Trying room {room_id}, instructor {instructor_id}: "
                          f"{len(available_times)} available time slots")

                for start_time in available_times:

                    # Score this assignment
                    assignment = {
                        'room_id': room_id,
                        'instructor_id': instructor_id,
                        'start_time': start_time,
                        'duration': duration,
                        'participants': participants
                    }

                    score = self._score_assignment(assignment, course_id)

                    if score > best_score:
                        best_score = score
                        best_assignment = assignment

                    # If using earliest available strategy, take first valid option
                    if (self.room_strategy == ResourceSelectionStrategy.EARLIEST_AVAILABLE and
                            self.instructor_strategy == ResourceSelectionStrategy.EARLIEST_AVAILABLE):
                        break

                if best_assignment and (
                        self.room_strategy == ResourceSelectionStrategy.EARLIEST_AVAILABLE or
                        self.instructor_strategy == ResourceSelectionStrategy.EARLIEST_AVAILABLE
                ):
                    break
        if self.verbose:
            if best_assignment:
                print(f"    ✅ Selected assignment for course {course_id}: "
                      f"room {best_assignment['room_id']}, "
                      f"instructor {best_assignment['instructor_id']}, "
                      f"time {best_assignment['start_time']}")
            else:
                print(f"    ❌ No valid assignment found for course {course_id}")

        return best_assignment

    def evaluate_solution(self, solution: Solution) -> float:
        """
        Evaluate solution and update statistics.
        Overridden to save best solution regardless of feasibility.
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

        # Update best solution regardless of feasibility
        if self.best_solution is None or objective > self.stats.best_objective:
            self.best_solution = solution  # Użyj referencji, nie kopii!
            self.stats.best_objective = objective
            self.stats.improvements += 1

            if self.verbose:
                feasible_str = " (feasible)" if is_feasible else " (infeasible)"
                print(f"  New best solution found: {objective:.2f}{feasible_str}")

        return objective

    def _get_course_order(self) -> List[int]:
        """Get ordered list of courses based on selection strategy."""
        courses = list(range(1, self.problem.N + 1))

        if self.course_strategy == CourseSelectionStrategy.RANDOM:
            random.shuffle(courses)
        else:
            # Sort by strategy-specific criteria
            courses.sort(key=lambda c: self._course_priority(c), reverse=True)

        return courses

    def _select_course_by_strategy(self, courses: List[int], solution: Solution) -> int:
        """Select course from list based on strategy."""
        if self.course_strategy == CourseSelectionStrategy.RANDOM:
            return random.choice(courses)

        # Find course with highest priority
        best_course = max(courses, key=lambda c: self._course_priority(c))
        return best_course

    def _course_priority(self, course_id: int) -> float:
        """Calculate priority score for course based on strategy."""
        course_idx = course_id - 1

        if self.course_strategy == CourseSelectionStrategy.PROFIT_DENSITY:
            return self.problem.get_course_profit_density(course_id)

        elif self.course_strategy == CourseSelectionStrategy.HIGHEST_PRICE:
            return self.problem.course_prices[course_idx]

        elif self.course_strategy == CourseSelectionStrategy.LONGEST_DURATION:
            return self.problem.course_durations[course_idx]

        elif self.course_strategy == CourseSelectionStrategy.SHORTEST_DURATION:
            return -self.problem.course_durations[course_idx]  # Negative for ascending order

        elif self.course_strategy == CourseSelectionStrategy.MOST_PARTICIPANTS:
            return self.problem.course_max_participants[course_idx]

        elif self.course_strategy == CourseSelectionStrategy.MOST_CONSTRAINED:
            # Count available options (fewer = higher priority)
            available_instructors = len(self.problem.get_available_instructors_for_course(course_id, 1))
            available_rooms = len(self.problem.get_available_rooms_for_course(course_id, 1))
            flexibility = available_instructors * available_rooms
            return -flexibility  # Negative because less flexibility = higher priority

        return 0.0

    def _determine_participants(self, course_id: int) -> int:
        """Determine number of participants for course."""
        course_idx = course_id - 1
        max_participants = self.problem.course_max_participants[course_idx]

        if self.participant_strategy == ParticipantStrategy.MAXIMUM:
            return max_participants

        elif self.participant_strategy == ParticipantStrategy.PERCENTAGE:
            return max(1, int(max_participants * self.participant_percentage))

        elif self.participant_strategy == ParticipantStrategy.FIXED:
            return min(self.fixed_participants, max_participants)

        elif self.participant_strategy == ParticipantStrategy.RANDOM:
            return random.randint(1, max_participants)

        return max_participants

    def _get_compatible_rooms(self, course_id: int, participants: int) -> List[int]:
        """Get rooms compatible with course and having sufficient capacity."""
        compatible = []
        course_idx = course_id - 1

        for room_id in range(1, self.problem.M + 1):
            room_idx = room_id - 1

            # Check room-course compatibility
            if self.problem.R_matrix[course_idx, room_idx] == 1:
                # Check capacity
                if self.problem.room_capacities[room_idx] >= participants:
                    compatible.append(room_id)

        return compatible

    def _get_compatible_instructors(self, course_id: int) -> List[int]:
        """Get instructors compatible with course."""
        course_idx = course_id - 1
        compatible = []

        for instructor_id in range(1, self.problem.K + 1):
            instructor_idx = instructor_id - 1

            if self.problem.I_matrix[instructor_idx, course_idx] == 1:
                compatible.append(instructor_id)

        return compatible

    def _order_rooms_by_strategy(self, rooms: List[int], course_id: int, participants: int) -> List[int]:
        """Order rooms according to selection strategy."""
        if self.room_strategy == ResourceSelectionStrategy.RANDOM:
            rooms_copy = rooms.copy()
            random.shuffle(rooms_copy)
            return rooms_copy

        def room_score(room_id: int) -> float:
            room_idx = room_id - 1

            if self.room_strategy == ResourceSelectionStrategy.CHEAPEST:
                # Prefer lower cost rooms
                return -self.problem.room_usage_costs[room_idx]

            elif self.room_strategy == ResourceSelectionStrategy.BEST_FIT_CAPACITY:
                # Prefer room with capacity closest to participants
                capacity = self.problem.room_capacities[room_idx]
                if capacity < participants:
                    return float('-inf')  # Can't use this room
                return -(capacity - participants)  # Smaller excess = better

            elif self.room_strategy == ResourceSelectionStrategy.HIGHEST_AVAILABILITY:
                # Count how many time slots this room is free
                # (This is approximate - doesn't account for current solution state)
                return self.problem.T  # Simplified - all rooms equally available initially

            return 0.0

        return sorted(rooms, key=room_score, reverse=True)

    def _order_instructors_by_strategy(self, instructors: List[int], course_id: int) -> List[int]:
        """Order instructors according to selection strategy."""
        if self.instructor_strategy == ResourceSelectionStrategy.RANDOM:
            instructors_copy = instructors.copy()
            random.shuffle(instructors_copy)
            return instructors_copy

        def instructor_score(instructor_id: int) -> float:
            instructor_idx = instructor_id - 1

            if self.instructor_strategy == ResourceSelectionStrategy.CHEAPEST:
                # Prefer lower cost instructors
                return -self.problem.instructor_hourly_rates[instructor_idx]

            elif self.instructor_strategy == ResourceSelectionStrategy.HIGHEST_AVAILABILITY:
                # Count available time slots
                return np.sum(self.problem.A_matrix[instructor_idx, :])

            return 0.0

        return sorted(instructors, key=instructor_score, reverse=True)

    def _find_available_times(self, solution: Solution, room_id: int,
                              instructor_id: int, duration: int) -> List[int]:
        """Find available time slots for room and instructor."""
        available_times = []
        instructor_idx = instructor_id - 1
        room_idx = room_id - 1

        for start_time in range(1, self.problem.T - duration + 2):
            can_assign = True

            # Check each required time slot
            for t_offset in range(duration):
                t_absolute = start_time + t_offset
                t_matrix = t_absolute - 1  # Convert to 0-indexed for matrices

                # 1. Check time horizon
                if t_absolute > self.problem.T:
                    can_assign = False
                    break

                # 2. Check instructor availability in A_matrix
                if self.problem.A_matrix[instructor_idx, t_matrix] == 0:
                    can_assign = False
                    break

                # 3. Check if instructor is free in solution
                if not solution.is_instructor_available(instructor_id, t_absolute):
                    can_assign = False
                    break

                # 4. Check if room is free in solution
                if not solution.is_room_available(room_id, t_absolute):
                    can_assign = False
                    break

            if can_assign:
                available_times.append(start_time)

                # If using earliest available strategy, take first option
                if (self.room_strategy == ResourceSelectionStrategy.EARLIEST_AVAILABLE and
                        self.instructor_strategy == ResourceSelectionStrategy.EARLIEST_AVAILABLE):
                    break

        return available_times

    def _score_assignment(self, assignment: Dict, course_id: int) -> float:
        """Score an assignment for selection purposes."""
        score = 0.0
        course_idx = course_id - 1
        room_idx = assignment['room_id'] - 1
        instructor_idx = assignment['instructor_id'] - 1

        # Revenue component
        revenue = self.problem.course_prices[course_idx] * assignment['participants']
        score += revenue

        # Cost components (negative)
        room_cost = self.problem.room_usage_costs[room_idx] * assignment['duration']
        instructor_cost = self.problem.instructor_hourly_rates[instructor_idx] * assignment['duration']
        score -= (room_cost + instructor_cost)

        # Preference for earlier time slots (slight bias)
        score += (self.problem.T - assignment['start_time']) * 0.1

        return score


def create_greedy_variants() -> List[Dict]:
    """
    Create different greedy algorithm variants with various parameter combinations.

    Returns:
        List of parameter dictionaries for different greedy variants
    """
    variants = []

    # Basic greedy variants
    variants.append({
        'name': 'Greedy_ProfitDensity_Cheapest',
        'course_selection_strategy': 'profit_density',
        'room_selection_strategy': 'cheapest',
        'instructor_selection_strategy': 'cheapest',
        'participant_strategy': 'maximum'
    })

    variants.append({
        'name': 'Greedy_HighestPrice_EarliestTime',
        'course_selection_strategy': 'highest_price',
        'room_selection_strategy': 'earliest_available',
        'instructor_selection_strategy': 'earliest_available',
        'participant_strategy': 'maximum'
    })

    variants.append({
        'name': 'Greedy_ShortestFirst_BestFit',
        'course_selection_strategy': 'shortest_duration',
        'room_selection_strategy': 'best_fit_capacity',
        'instructor_selection_strategy': 'cheapest',
        'participant_strategy': 'percentage',
        'participant_percentage': 0.9
    })

    variants.append({
        'name': 'Greedy_MostConstrained_Random',
        'course_selection_strategy': 'most_constrained',
        'room_selection_strategy': 'random',
        'instructor_selection_strategy': 'random',
        'participant_strategy': 'maximum',
        'random_seed': 42
    })

    # 5. "Business Optimal" - moja rekomendacja == nr 1.
    variants.append({
        'name': 'Greedy_BusinessOptimal',
        'course_selection_strategy': 'profit_density',
        'room_selection_strategy': 'cheapest',
        'instructor_selection_strategy': 'cheapest',
        'participant_strategy': 'maximum'
    })

    # 6. "Balanced"
    variants.append({
        'name': 'Greedy_Balanced',
        'course_selection_strategy': 'profit_density',
        'room_selection_strategy': 'best_fit_capacity',
        'instructor_selection_strategy': 'highest_availability',
        'participant_strategy': 'percentage',
        'participant_percentage': 0.9
    })

    # 7. "Fast Fill"
    variants.append({
        'name': 'Greedy_FastFill',
        'course_selection_strategy': 'shortest_duration',
        'room_selection_strategy': 'earliest_available',
        'instructor_selection_strategy': 'earliest_available',
        'participant_strategy': 'percentage',
        'participant_percentage': 0.7
    })

    # 8. "Smart Constrained"
    variants.append({
        'name': 'Greedy_SmartConstrained',
        'course_selection_strategy': 'most_constrained',
        'room_selection_strategy': 'best_fit_capacity',
        'instructor_selection_strategy': 'cheapest',
        'participant_strategy': 'maximum'
    })

    # 9. Wszystko losowe (baseline)
    variants.append({
        'name': 'Greedy_AllRandom',
        'course_selection_strategy': 'random',
        'room_selection_strategy': 'random',
        'instructor_selection_strategy': 'random',
        'participant_strategy': 'random',
        'random_seed': 123
    })

    # 10. Maksymalny koszt (worst case)
    variants.append({
        'name': 'Greedy_ExpensiveTest',
        'course_selection_strategy': 'longest_duration',
        'room_selection_strategy': 'highest_availability',  # może wybierać drogie
        'instructor_selection_strategy': 'highest_availability',
        'participant_strategy': 'fixed',
        'fixed_participants': 5  # małe grupy
    })

    return variants


if __name__ == "__main__":
    # Test greedy algorithm
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

        print("Testing Greedy Algorithm variants...")

        # Test different variants
        variants = create_greedy_variants()

        for variant in variants[:10]:  # Test first 2 variants
            print(f"\n{'=' * 50}")
            print(f"Testing: {variant['name']}")
            print(f"{'=' * 50}")

            # Create and run algorithm
            greedy = GreedyAlgorithm(problem, verbose=True, **variant)
            solution = greedy.run()

            # Print results
            greedy.print_statistics()

            # Validate solution
            x, y, u = solution.to_arrays()
            is_feasible, violations = problem.validate_solution(x, y, u)

            print(f"\nSolution validation:")
            print(f"  Feasible: {is_feasible}")
            if violations:
                print(f"  Violations: {len(violations)}")
                for violation in violations[:5]:  # Show first 5
                    print(f"    - {violation}")

        print(f"\n✓ Greedy algorithm testing completed!")

    except Exception as e:
        print(f"Error testing greedy algorithm: {e}")
        import traceback

        traceback.print_exc()