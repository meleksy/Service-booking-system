import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from copy import deepcopy


class Solution:
    """
    Represents a solution to the course scheduling problem.

    Encapsulates the decision variables:
    - x[i,m,t]: course i in room m at time t
    - y[i,k,t]: course i taught by instructor k at time t
    - u[i]: number of participants in course i

    Provides methods for easy manipulation and querying of assignments.
    """

    def __init__(self, N: int, M: int, K: int, T: int, problem=None):
        """
        Initialize empty solution with given dimensions.

        Args:
            N: Number of courses
            M: Number of rooms
            K: Number of instructors
            T: Time horizon
            problem: CourseSchedulingProblem instance for validation
        """
        self.N = N
        self.M = M
        self.K = K
        self.T = T
        self.problem = problem

        # Decision variables - initialized as zeros
        self.x = np.zeros((N, M, T), dtype=int)  # Course-room-time assignments
        self.y = np.zeros((N, K, T), dtype=int)  # Course-instructor-time assignments
        self.u = np.zeros(N, dtype=int)  # Number of participants per course

        # Cache for objective value and feasibility (invalidated on changes)
        self._objective_value = None
        self._is_feasible = None
        self._violations = None

        # Track which assignments are made for quick queries
        self._scheduled_courses = set()  # Set of course indices that are scheduled

    def copy(self) -> 'Solution':
        """Create deep copy of this solution."""
        new_solution = Solution(self.N, self.M, self.K, self.T, self.problem)
        new_solution.x = self.x.copy()
        new_solution.y = self.y.copy()
        new_solution.u = self.u.copy()
        new_solution._scheduled_courses = self._scheduled_courses.copy()
        # Don't copy cached values - let them be recalculated
        return new_solution

    def clear(self) -> None:
        """Reset solution to empty state."""
        self.x.fill(0)
        self.y.fill(0)
        self.u.fill(0)
        self._scheduled_courses.clear()
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate cached values when solution changes."""
        self._objective_value = None
        self._is_feasible = None
        self._violations = None

    # DODAJ DEBUG DO assign_course() w solution.py

    def assign_course(self, course_id: int, room_id: int, instructor_id: int,
                      start_time: int, duration: int, participants: int) -> bool:
        """
        Assign a course to room, instructor and time slots.
        """
        # Convert to 0-indexed
        i = course_id - 1
        m = room_id - 1
        k = instructor_id - 1
        t_start = start_time - 1

        print(
            f"🔧 DEBUG assign_course: course={course_id}, room={room_id}, instructor={instructor_id}, time={start_time}, duration={duration}")
        print(f"   Converted indices: i={i}, m={m}, k={k}, t_start={t_start}")

        # Validate indices
        if not (0 <= i < self.N and 0 <= m < self.M and 0 <= k < self.K):
            print(f"   ❌ Invalid indices: N={self.N}, M={self.M}, K={self.K}")
            return False
        if not (0 <= t_start < self.T and t_start + duration <= self.T):
            print(f"   ❌ Invalid time: T={self.T}, t_start={t_start}, duration={duration}")
            return False

        print(f"   ✅ Indices valid")

        # Check for conflicts
        for t_offset in range(duration):
            t = t_start + t_offset

            # Room conflict
            room_occupied = np.sum(self.x[:, m, t])
            if room_occupied > 0:
                print(
                    f"   ❌ Room conflict at time {t + 1}: room {room_id} occupied by {np.where(self.x[:, m, t] == 1)[0] + 1}")
                return False

            # Instructor conflict
            instructor_occupied = np.sum(self.y[:, k, t])
            if instructor_occupied > 0:
                print(
                    f"   ❌ Instructor conflict at time {t + 1}: instructor {instructor_id} teaching {np.where(self.y[:, k, t] == 1)[0] + 1}")
                return False

        print(f"   ✅ No conflicts found")

        # Make the assignment
        print(f"   📝 Making assignment...")
        for t_offset in range(duration):
            t = t_start + t_offset
            print(f"      Setting x[{i}, {m}, {t}] = 1")
            print(f"      Setting y[{i}, {k}, {t}] = 1")
            self.x[i, m, t] = 1
            self.y[i, k, t] = 1

        print(f"   📝 Setting u[{i}] = {participants}")
        self.u[i] = participants

        print(f"   📝 Adding {i} to _scheduled_courses")
        self._scheduled_courses.add(i)

        print(f"   📝 Invalidating cache")
        self._invalidate_cache()

        # Verify assignment was made
        x_sum = np.sum(self.x[i, :, :])
        y_sum = np.sum(self.y[i, :, :])
        print(f"   🔍 Verification: x_sum={x_sum}, y_sum={y_sum}, u[{i}]={self.u[i]}")
        print(f"   🔍 _scheduled_courses: {self._scheduled_courses}")

        print(f"   ✅ Assignment completed successfully")
        return True


    def remove_course_assignment(self, course_id: int) -> None:
        """
        Remove all assignments for a given course.

        Args:
            course_id: Course ID (1-indexed)
        """
        i = course_id - 1
        if 0 <= i < self.N:
            self.x[i, :, :] = 0
            self.y[i, :, :] = 0
            self.u[i] = 0
            self._scheduled_courses.discard(i)
            self._invalidate_cache()

    def get_course_assignment(self, course_id: int) -> Optional[Dict]:
        """Get assignment details for a course."""
        i = course_id - 1
        if i not in self._scheduled_courses:  # ← TUTAJ MOŻE BYĆ PROBLEM!
            return None

        # Find room assignment
        room_assignments = np.where(self.x[i, :, :] == 1)
        if len(room_assignments[0]) == 0:
            return None

        room_id = room_assignments[0][0] + 1  # Convert to 1-indexed
        time_slots = room_assignments[1] + 1  # Convert to 1-indexed

        # Find instructor assignment
        instructor_assignments = np.where(self.y[i, :, :] == 1)
        instructor_id = instructor_assignments[0][0] + 1 if len(instructor_assignments[0]) > 0 else None

        return {
            'course_id': course_id,
            'room_id': room_id,
            'instructor_id': instructor_id,
            'start_time': int(np.min(time_slots)),
            'end_time': int(np.max(time_slots)),
            'duration': len(time_slots),
            'participants': int(self.u[i])
        }

    def get_all_assignments(self) -> List[Dict]:
        """Get list of all course assignments."""
        assignments = []
        for course_id in range(1, self.N + 1):
            assignment = self.get_course_assignment(course_id)
            if assignment:
                assignments.append(assignment)
        return assignments

    def get_scheduled_courses(self) -> Set[int]:
        """Get set of scheduled course IDs (1-indexed)."""
        return {i + 1 for i in self._scheduled_courses}

    def get_unscheduled_courses(self) -> Set[int]:
        """Get set of unscheduled course IDs (1-indexed)."""
        all_courses = set(range(1, self.N + 1))
        return all_courses - self.get_scheduled_courses()

    def is_room_available(self, room_id: int, time_slot: int) -> bool:
        """Check if room is available at given time slot."""
        m = room_id - 1
        t = time_slot - 1
        if 0 <= m < self.M and 0 <= t < self.T:
            return np.sum(self.x[:, m, t]) == 0
        return False

    def is_instructor_available(self, instructor_id: int, time_slot: int) -> bool:
        """Check if instructor is available at given time slot."""
        k = instructor_id - 1
        t = time_slot - 1
        if 0 <= k < self.K and 0 <= t < self.T:
            return np.sum(self.y[:, k, t]) == 0
        return False

    def get_room_schedule(self, room_id: int) -> List[Tuple[int, int]]:
        """
        Get schedule for a room.

        Args:
            room_id: Room ID (1-indexed)

        Returns:
            List of (course_id, time_slot) tuples for occupied slots
        """
        m = room_id - 1
        schedule = []

        if 0 <= m < self.M:
            for t in range(self.T):
                for i in range(self.N):
                    if self.x[i, m, t] == 1:
                        schedule.append((i + 1, t + 1))  # Convert to 1-indexed

        return sorted(schedule, key=lambda x: x[1])  # Sort by time

    def get_instructor_schedule(self, instructor_id: int) -> List[Tuple[int, int]]:
        """
        Get schedule for an instructor.

        Args:
            instructor_id: Instructor ID (1-indexed)

        Returns:
            List of (course_id, time_slot) tuples for teaching slots
        """
        k = instructor_id - 1
        schedule = []

        if 0 <= k < self.K:
            for t in range(self.T):
                for i in range(self.N):
                    if self.y[i, k, t] == 1:
                        schedule.append((i + 1, t + 1))  # Convert to 1-indexed

        return sorted(schedule, key=lambda x: x[1])  # Sort by time

    def count_conflicts(self) -> Dict[str, int]:
        """
        Count different types of constraint violations.

        Returns:
            Dictionary with conflict counts
        """
        conflicts = {
            'room_conflicts': 0,
            'instructor_conflicts': 0,
            'unassigned_courses': 0,
            'duration_violations': 0
        }

        # Room conflicts (multiple courses in same room/time)
        for m in range(self.M):
            for t in range(self.T):
                if np.sum(self.x[:, m, t]) > 1:
                    conflicts['room_conflicts'] += 1

        # Instructor conflicts (multiple courses for same instructor/time)
        for k in range(self.K):
            for t in range(self.T):
                if np.sum(self.y[:, k, t]) > 1:
                    conflicts['instructor_conflicts'] += 1

        # Unassigned courses
        conflicts['unassigned_courses'] = self.N - len(self._scheduled_courses)

        # Duration violations (courses with wrong number of time slots)
        # Note: This requires problem instance to check expected durations
        # For now, just count courses with inconsistent x and y assignments
        for i in range(self.N):
            x_count = np.sum(self.x[i, :, :])
            y_count = np.sum(self.y[i, :, :])
            if x_count != y_count:
                conflicts['duration_violations'] += 1

        return conflicts

    def calculate_utilization(self) -> Dict[str, float]:
        """
        Calculate resource utilization statistics.

        Returns:
            Dictionary with utilization percentages
        """
        total_room_slots = self.M * self.T
        used_room_slots = np.sum(self.x)

        total_instructor_slots = self.K * self.T
        used_instructor_slots = np.sum(self.y)

        return {
            'room_utilization': used_room_slots / total_room_slots if total_room_slots > 0 else 0.0,
            'instructor_utilization': used_instructor_slots / total_instructor_slots if total_instructor_slots > 0 else 0.0,
            'courses_scheduled': len(self._scheduled_courses) / self.N if self.N > 0 else 0.0
        }

    def swap_courses(self, course1_id: int, course2_id: int) -> bool:
        """
        Swap assignments of two courses (useful for local search).

        Args:
            course1_id, course2_id: Course IDs (1-indexed)

        Returns:
            True if swap successful, False if either course not assigned
        """
        assignment1 = self.get_course_assignment(course1_id)
        assignment2 = self.get_course_assignment(course2_id)

        if not assignment1 or not assignment2:
            return False

        # Remove both assignments
        self.remove_course_assignment(course1_id)
        self.remove_course_assignment(course2_id)

        # Swap assignments
        success1 = self.assign_course(
            course1_id, assignment2['room_id'], assignment2['instructor_id'],
            assignment2['start_time'], assignment2['duration'], assignment1['participants']
        )

        success2 = self.assign_course(
            course2_id, assignment1['room_id'], assignment1['instructor_id'],
            assignment1['start_time'], assignment1['duration'], assignment2['participants']
        )

        if not (success1 and success2):
            # Restore original assignments if swap failed
            self.assign_course(
                course1_id, assignment1['room_id'], assignment1['instructor_id'],
                assignment1['start_time'], assignment1['duration'], assignment1['participants']
            )
            self.assign_course(
                course2_id, assignment2['room_id'], assignment2['instructor_id'],
                assignment2['start_time'], assignment2['duration'], assignment2['participants']
            )
            return False

        return True

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get solution as raw NumPy arrays.

        Returns:
            Tuple of (x, y, u) arrays
        """
        return self.x.copy(), self.y.copy(), self.u.copy()

    def from_arrays(self, x: np.ndarray, y: np.ndarray, u: np.ndarray) -> None:
        """
        Set solution from NumPy arrays.

        Args:
            x: Course-room-time assignments [N x M x T]
            y: Course-instructor-time assignments [N x K x T]
            u: Participants per course [N]
        """
        if x.shape != (self.N, self.M, self.T):
            raise ValueError(f"x shape {x.shape} != expected {(self.N, self.M, self.T)}")
        if y.shape != (self.N, self.K, self.T):
            raise ValueError(f"y shape {y.shape} != expected {(self.N, self.K, self.T)}")
        if u.shape != (self.N,):
            raise ValueError(f"u shape {u.shape} != expected {(self.N,)}")

        self.x = x.copy()
        self.y = y.copy()
        self.u = u.copy()

        # Rebuild scheduled courses set
        self._scheduled_courses = set()
        for i in range(self.N):
            if np.sum(self.x[i, :, :]) > 0:
                self._scheduled_courses.add(i)

        self._invalidate_cache()

    def print_solution_summary(self) -> None:
        """Print human-readable summary of the solution."""
        print(f"\n=== Solution Summary ===")
        print(f"Scheduled courses: {len(self._scheduled_courses)}/{self.N}")

        utilization = self.calculate_utilization()
        print(f"Room utilization: {utilization['room_utilization']:.1%}")
        print(f"Instructor utilization: {utilization['instructor_utilization']:.1%}")

        conflicts = self.count_conflicts()
        total_conflicts = sum(conflicts.values())
        print(f"Total conflicts: {total_conflicts}")
        if total_conflicts > 0:
            for conflict_type, count in conflicts.items():
                if count > 0:
                    print(f"  {conflict_type}: {count}")

        print(f"\nCourse assignments:")
        assignments = self.get_all_assignments()
        for assignment in sorted(assignments, key=lambda x: x['course_id']):
            print(f"  Course {assignment['course_id']}: "
                  f"Room {assignment['room_id']}, "
                  f"Instructor {assignment['instructor_id']}, "
                  f"Time {assignment['start_time']}-{assignment['end_time']}, "
                  f"{assignment['participants']} participants")

        unscheduled = self.get_unscheduled_courses()
        if unscheduled:
            print(f"\nUnscheduled courses: {sorted(unscheduled)}")


def create_empty_solution(N: int, M: int, K: int, T: int, problem=None) -> Solution:
    """
    Create empty solution with given dimensions.

    Args:
        N, M, K, T: Problem dimensions
        problem: CourseSchedulingProblem instance

    Returns:
        Empty Solution object
    """
    return Solution(N, M, K, T, problem)


def create_solution_from_arrays(x: np.ndarray, y: np.ndarray, u: np.ndarray) -> Solution:
    """
    Create solution from NumPy arrays.

    Args:
        x, y, u: Solution arrays

    Returns:
        Solution object
    """
    N, M, T = x.shape
    K = y.shape[1]

    solution = Solution(N, M, K, T)
    solution.from_arrays(x, y, u)

    return solution


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Solution class...")

    # Create empty solution
    solution = Solution(N=5, M=3, K=3, T=20)

    # Test assignment
    success = solution.assign_course(
        course_id=1, room_id=1, instructor_id=1,
        start_time=1, duration=2, participants=15
    )
    print(f"Assignment successful: {success}")

    # Test another assignment
    success = solution.assign_course(
        course_id=2, room_id=2, instructor_id=2,
        start_time=5, duration=1, participants=12
    )
    print(f"Second assignment successful: {success}")

    # Print summary
    solution.print_solution_summary()

    # Test queries
    print(f"\nRoom 1 available at time 3? {solution.is_room_available(1, 3)}")
    print(f"Instructor 1 available at time 5? {solution.is_instructor_available(1, 5)}")

    # Test copy
    solution_copy = solution.copy()
    print(f"Copy created with {len(solution_copy.get_scheduled_courses())} scheduled courses")