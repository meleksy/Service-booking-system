import numpy as np
from typing import List, Dict, Tuple, Optional
import sys
import os

# Import DataLoader from utils
try:
    from ..utils.data_loader import DataLoader
except ImportError:
    # Fallback for direct execution
    import sys
    import os

    utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
    sys.path.append(utils_path)
    from utils.data_loader import DataLoader


class CourseSchedulingProblem:
    """
    Represents the course scheduling optimization problem.

    This class encapsulates all problem data, constraints, and provides methods
    to evaluate solutions according to the mathematical model defined in the PDF.

    Mathematical model variables:
    - x[i,m,t] ∈ {0,1}: course i in room m at time t
    - y[i,k,t] ∈ {0,1}: course i taught by instructor k at time t
    - u[i] ≥ 0: number of participants in course i
    """

    def __init__(self, data_loader: DataLoader):
        """
        Initialize problem instance from loaded data.

        Args:
            data_loader: DataLoader object with loaded instance data
        """
        self.data_loader = data_loader

        # Get instance name from data_loader
        self.instance_name = getattr(data_loader, 'instance_name', 'unknown_instance')

        # Problem dimensions
        self.N = data_loader.N  # number of courses
        self.M = data_loader.M  # number of rooms
        self.K = data_loader.K  # number of instructors
        self.T = data_loader.T  # time horizon

        # Compatibility matrices
        self.I_matrix = data_loader.I_matrix  # [K x N] instructor-course compatibility
        self.R_matrix = data_loader.R_matrix  # [N x M] course-room compatibility
        self.A_matrix = data_loader.A_matrix  # [K x T] instructor availability

        # Course parameters
        self.course_prices = np.array([
            data_loader.courses.iloc[i]['price_per_participant']
            for i in range(self.N)
        ])
        self.course_durations = np.array([
            data_loader.courses.iloc[i]['duration']
            for i in range(self.N)
        ])
        self.course_max_participants = np.array([
            data_loader.courses.iloc[i]['max_participants']
            for i in range(self.N)
        ])

        # Room parameters
        self.room_usage_costs = np.array([
            data_loader.rooms.iloc[m]['usage_cost_per_hour']
            for m in range(self.M)
        ])
        self.room_idle_costs = np.array([
            data_loader.rooms.iloc[m]['idle_cost_per_hour']
            for m in range(self.M)
        ])
        self.room_capacities = np.array([
            data_loader.rooms.iloc[m]['capacity']
            for m in range(self.M)
        ])

        # Instructor parameters
        self.instructor_hourly_rates = np.array([
            data_loader.instructors.iloc[k]['hourly_rate']
            for k in range(self.K)
        ])

        # Course participants (actual enrollments)
        self.course_participants = np.array([
            data_loader.get_course_participants(i + 1)  # Convert to 1-indexed
            for i in range(self.N)
        ])

        print(f"Loaded course participants: {self.course_participants}")

        print(f"Problem initialized: {self.N} courses, {self.M} rooms, {self.K} instructors, {self.T} time slots")

    def calculate_objective(self, x: np.ndarray, y: np.ndarray, u: np.ndarray) -> float:
        """
        Calculate objective function value according to the mathematical model.

        Objective: Max Z = Σ(p_i * u_i) - Σ(c_m * Σx_imt) - Σ(h_m * (1-Σx_imt)) - Σ(c_k * Σy_ikt)

        Args:
            x: [N x M x T] binary array - course i in room m at time t
            y: [N x K x T] binary array - course i taught by instructor k at time t
            u: [N] array - number of participants in course i

        Returns:
            Objective function value (higher is better)
        """
        # First term: Revenue from courses
        revenue = np.sum(self.course_prices * self.course_participants)

        # Second term: Room usage costs
        room_usage_cost = 0.0
        for m in range(self.M):
            for t in range(self.T):
                if np.sum(x[:, m, t]) > 0:  # Room m used at time t
                    room_usage_cost += self.room_usage_costs[m]

        # Third term: Room idle costs
        room_idle_cost = 0.0
        for m in range(self.M):
            for t in range(self.T):
                if np.sum(x[:, m, t]) == 0:  # Room m idle at time t
                    room_idle_cost += self.room_idle_costs[m]

        # Fourth term: Instructor costs
        instructor_cost = 0.0
        for k in range(self.K):
            for t in range(self.T):
                if np.sum(y[:, k, t]) > 0:  # Instructor k working at time t
                    instructor_cost += self.instructor_hourly_rates[k]

        objective_value = revenue - room_usage_cost - room_idle_cost - instructor_cost

        return objective_value

    def calculate_detailed_objective(self, x: np.ndarray, y: np.ndarray, u: np.ndarray) -> Dict[str, float]:
        """
        Calculate detailed breakdown of objective function components.

        Returns:
            Dictionary with detailed cost/revenue breakdown
        """
        # First term: Revenue from courses
        if hasattr(self, 'course_participants'):
            # Use actual participants if available
            revenue = float(np.sum(self.course_prices * self.course_participants))
        else:
            # Use u parameter
            revenue = float(np.sum(self.course_prices * u))

        # Second term: Room usage costs
        room_usage_cost = 0.0
        for m in range(self.M):
            for t in range(self.T):
                if np.sum(x[:, m, t]) > 0:  # Room m used at time t
                    room_usage_cost += self.room_usage_costs[m]

        # Third term: Room idle costs
        room_idle_cost = 0.0
        for m in range(self.M):
            for t in range(self.T):
                if np.sum(x[:, m, t]) == 0:  # Room m idle at time t
                    room_idle_cost += self.room_idle_costs[m]

        # Fourth term: Instructor costs
        instructor_cost = 0.0
        for k in range(self.K):
            for t in range(self.T):
                if np.sum(y[:, k, t]) > 0:  # Instructor k working at time t
                    instructor_cost += self.instructor_hourly_rates[k]

        # Calculate total objective
        total_objective = revenue - room_usage_cost - room_idle_cost - instructor_cost

        return {
            'zysk': float(revenue),
            'koszt_uzytkowania': float(room_usage_cost),
            'koszt_niewykorzystania': float(room_idle_cost),
            'koszt_instruktorow': float(instructor_cost),
            'funkcja': float(total_objective)
        }

    def validate_solution(self, x: np.ndarray, y: np.ndarray, u: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validate solution against all constraints.

        Args:
            x: [N x M x T] binary array - course assignments to rooms
            y: [N x K x T] binary array - course assignments to instructors
            u: [N] array - number of participants

        Returns:
            Tuple of (is_feasible, list_of_violations)
        """
        violations = []

        # Check array shapes
        if x.shape != (self.N, self.M, self.T):
            violations.append(f"x array shape {x.shape} != expected {(self.N, self.M, self.T)}")
        if y.shape != (self.N, self.K, self.T):
            violations.append(f"y array shape {y.shape} != expected {(self.N, self.K, self.T)}")
        if u.shape != (self.N,):
            violations.append(f"u array shape {u.shape} != expected {(self.N,)}")

        if violations:  # If shape errors, can't continue validation
            return False, violations

        # Constraint 1: Each course assigned to exactly one room-time combination
        violations.extend(self._check_course_room_assignment(x))

        # Constraint 2: Room capacity constraints
        violations.extend(self._check_room_capacity(x, u))

        # Constraint 3: Room occupancy - only one course per room per time
        violations.extend(self._check_room_occupancy(x))

        # Constraint 4: Instructor constraints
        violations.extend(self._check_instructor_constraints(y))

        # Constraint 5: Course-instructor assignment consistency
        violations.extend(self._check_course_instructor_assignment(x, y))

        # Constraint 6: Equipment/compatibility constraints
        violations.extend(self._check_compatibility_constraints(x, y))

        # Constraint 7: Course duration constraints
        violations.extend(self._check_course_duration(x))

        # Additional checks
        violations.extend(self._check_participant_limits(u))
        violations.extend(self._check_binary_constraints(x, y))

        is_feasible = len(violations) == 0
        return is_feasible, violations

    def _check_course_room_assignment(self, x: np.ndarray) -> List[str]:
        """Check that each course is assigned to exactly one room-time combination."""
        violations = []

        for i in range(self.N):
            total_assignments = np.sum(x[i, :, :])
            expected_assignments = self.course_durations[i]

            if total_assignments != expected_assignments:
                violations.append(
                    f"Course {i + 1}: assigned {total_assignments} time slots, "
                    f"but duration is {expected_assignments}"
                )

        return violations

    def _check_room_capacity(self, x: np.ndarray, u: np.ndarray) -> List[str]:
        """Check room capacity constraints."""
        violations = []

        for i in range(self.N):
            for m in range(self.M):
                for t in range(self.T):
                    if x[i, m, t] == 1:
                        if u[i] > self.room_capacities[m]:
                            violations.append(
                                f"Course {i + 1} with {u[i]} participants "
                                f"exceeds room {m + 1} capacity {self.room_capacities[m]}"
                            )

        return violations

    def _check_room_occupancy(self, x: np.ndarray) -> List[str]:
        """Check that each room hosts at most one course at any time."""
        violations = []

        for m in range(self.M):
            for t in range(self.T):
                courses_in_room = np.sum(x[:, m, t])
                if courses_in_room > 1:
                    course_list = [i + 1 for i in range(self.N) if x[i, m, t] == 1]
                    violations.append(
                        f"Room {m + 1} at time {t + 1}: multiple courses {course_list}"
                    )

        return violations

    def _check_instructor_constraints(self, y: np.ndarray) -> List[str]:
        """Check instructor availability and single assignment constraints."""
        violations = []

        # Check instructor can only teach one course at a time
        for k in range(self.K):
            for t in range(self.T):
                courses_taught = np.sum(y[:, k, t])
                if courses_taught > 1:
                    course_list = [i + 1 for i in range(self.N) if y[i, k, t] == 1]
                    violations.append(
                        f"Instructor {k + 1} at time {t + 1}: teaching multiple courses {course_list}"
                    )

                # Check instructor availability
                if courses_taught > 0 and self.A_matrix[k, t] == 0:
                    violations.append(
                        f"Instructor {k + 1} assigned at time {t + 1} but not available"
                    )

        return violations

    def _check_course_instructor_assignment(self, x: np.ndarray, y: np.ndarray) -> List[str]:
        """Check that each scheduled course has exactly one instructor."""
        violations = []

        for i in range(self.N):
            for t in range(self.T):
                # If course i is scheduled at time t (in any room)
                if np.sum(x[i, :, t]) > 0:
                    # It must have exactly one instructor
                    instructors_assigned = np.sum(y[i, :, t])
                    if instructors_assigned != 1:
                        violations.append(
                            f"Course {i + 1} at time {t + 1}: {instructors_assigned} instructors "
                            f"assigned (should be 1)"
                        )
                else:
                    # If course not scheduled, should have no instructors
                    instructors_assigned = np.sum(y[i, :, t])
                    if instructors_assigned > 0:
                        violations.append(
                            f"Course {i + 1} at time {t + 1}: not scheduled but has "
                            f"{instructors_assigned} instructors"
                        )

        return violations

    def _check_compatibility_constraints(self, x: np.ndarray, y: np.ndarray) -> List[str]:
        """Check instructor-course and room-course compatibility."""
        violations = []

        # Check instructor-course compatibility
        for i in range(self.N):
            for k in range(self.K):
                for t in range(self.T):
                    if y[i, k, t] == 1 and self.I_matrix[k, i] == 0:
                        violations.append(
                            f"Instructor {k + 1} assigned to course {i + 1} at time {t + 1} "
                            f"but not qualified"
                        )

        # Check room-course compatibility
        for i in range(self.N):
            for m in range(self.M):
                for t in range(self.T):
                    if x[i, m, t] == 1 and self.R_matrix[i, m] == 0:
                        violations.append(
                            f"Course {i + 1} assigned to room {m + 1} at time {t + 1} "
                            f"but room doesn't meet requirements"
                        )

        return violations

    def _check_course_duration(self, x: np.ndarray) -> List[str]:
        """Check that courses are scheduled for consecutive time slots."""
        violations = []

        for i in range(self.N):
            duration = self.course_durations[i]

            # Find all time slots where course i is scheduled
            scheduled_times = []
            for m in range(self.M):
                for t in range(self.T):
                    if x[i, m, t] == 1:
                        scheduled_times.append(t)

            if len(scheduled_times) != duration:
                continue  # This will be caught by course assignment check

            # Check if times are consecutive
            if len(scheduled_times) > 1:
                scheduled_times.sort()
                for j in range(1, len(scheduled_times)):
                    if scheduled_times[j] != scheduled_times[j - 1] + 1:
                        violations.append(
                            f"Course {i + 1}: non-consecutive time slots {scheduled_times}"
                        )
                        break

        return violations

    def _check_participant_limits(self, u: np.ndarray) -> List[str]:
        """Check participant number constraints."""
        violations = []

        for i in range(self.N):
            if u[i] < 0:
                violations.append(f"Course {i + 1}: negative participants {u[i]}")
            elif u[i] > self.course_max_participants[i]:
                violations.append(
                    f"Course {i + 1}: {u[i]} participants exceeds maximum "
                    f"{self.course_max_participants[i]}"
                )

        return violations

    def _check_binary_constraints(self, x: np.ndarray, y: np.ndarray) -> List[str]:
        """Check that x and y are binary."""
        violations = []

        if not np.all(np.isin(x, [0, 1])):
            violations.append("x array contains non-binary values")

        if not np.all(np.isin(y, [0, 1])):
            violations.append("y array contains non-binary values")

        return violations

    def is_feasible(self, x: np.ndarray, y: np.ndarray, u: np.ndarray) -> bool:
        """Quick feasibility check without detailed violation list."""
        feasible, _ = self.validate_solution(x, y, u)
        return feasible

    def get_course_profit_density(self, course_id: int) -> float:
        """Get profit per hour for a course (useful for greedy algorithms)."""
        i = course_id - 1
        max_revenue = self.course_prices[i] * self.course_max_participants[i]
        duration = self.course_durations[i]
        return max_revenue / duration if duration > 0 else 0.0

    def get_available_instructors_for_course(self, course_id: int, time_slot: int) -> List[int]:
        """Get list of available and qualified instructors for a course at given time."""
        i = course_id - 1
        t = time_slot - 1

        available_instructors = []
        for k in range(self.K):
            if self.I_matrix[k, i] == 1 and self.A_matrix[k, t] == 1:
                available_instructors.append(k + 1)  # Convert to 1-indexed

        return available_instructors

    def get_available_rooms_for_course(self, course_id: int, num_participants: int) -> List[int]:
        """Get list of suitable rooms for a course with given number of participants."""
        i = course_id - 1

        suitable_rooms = []
        for m in range(self.M):
            if (self.R_matrix[i, m] == 1 and
                    self.room_capacities[m] >= num_participants):
                suitable_rooms.append(m + 1)  # Convert to 1-indexed

        return suitable_rooms

    def get_actual_participants(self, course_id: int) -> int:
        """Get actual number of participants for a course."""
        course_idx = course_id - 1
        if 0 <= course_idx < len(self.course_participants):
            return self.course_participants[course_idx]
        return 0

    def print_problem_summary(self):
        """Print summary of problem parameters."""
        print(f"\n=== Problem Summary ===")
        print(f"Courses: {self.N}, Rooms: {self.M}, Instructors: {self.K}, Time slots: {self.T}")

        print(f"\nCourse parameters:")
        for i in range(self.N):
            print(f"  Course {i + 1}: duration={self.course_durations[i]}, "
                  f"price={self.course_prices[i]:.2f}, "
                  f"max_participants={self.course_max_participants[i]}")

        print(f"\nRoom parameters:")
        for m in range(self.M):
            print(f"  Room {m + 1}: capacity={self.room_capacities[m]}, "
                  f"usage_cost={self.room_usage_costs[m]:.2f}, "
                  f"idle_cost={self.room_idle_costs[m]:.2f}")

        print(f"\nInstructor parameters:")
        for k in range(self.K):
            available_slots = np.sum(self.A_matrix[k, :])
            print(f"  Instructor {k + 1}: rate={self.instructor_hourly_rates[k]:.2f}, "
                  f"available_slots={available_slots}")


def create_problem_from_instance(instance_path: str) -> CourseSchedulingProblem:
    """
    Convenience function to create problem from instance path.

    Args:
        instance_path: Path to instance directory

    Returns:
        CourseSchedulingProblem object
    """
    # Import here to avoid circular imports
    utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
    sys.path.append(utils_path)
    from utils.data_loader import load_instance

    loader = load_instance(instance_path)
    problem = CourseSchedulingProblem(loader)

    return problem


if __name__ == "__main__":
    # Example usage and testing
    try:
        # Use absolute path from project root
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        instance_path = os.path.join(project_root, "data", "instances", "small", "small_01")

        problem = create_problem_from_instance(instance_path)
        problem.print_problem_summary()

        # Create a simple test solution
        x = np.zeros((problem.N, problem.M, problem.T), dtype=int)
        y = np.zeros((problem.N, problem.K, problem.T), dtype=int)
        u = np.array([10, 8, 15, 6, 20])  # Sample participant numbers

        # Assign course 1 to room 1, instructor 1, time slots 1-2
        x[0, 0, 0] = 1  # Course 1, Room 1, Time 1
        x[0, 0, 1] = 1  # Course 1, Room 1, Time 2
        y[0, 0, 0] = 1  # Course 1, Instructor 1, Time 1
        y[0, 0, 1] = 1  # Course 1, Instructor 1, Time 2

        # Validate solution
        is_feasible, violations = problem.validate_solution(x, y, u)
        print(f"\nTest solution feasible: {is_feasible}")
        if violations:
            print("Violations:")
            for violation in violations:
                print(f"  - {violation}")

        # Calculate objective
        obj_value = problem.calculate_objective(x, y, u)
        print(f"Objective value: {obj_value:.2f}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the test instance created and data_loader.py available")
