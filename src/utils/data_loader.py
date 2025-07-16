import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Tuple, Optional


class DataLoader:
    """
    Loads course scheduling problem instances from CSV files and converts them
    to NumPy matrices for efficient computation in optimization algorithms.
    """

    def __init__(self):
        # Basic data DataFrames
        self.courses = None
        self.rooms = None
        self.instructors = None
        self.metadata = None

        # Compatibility and availability matrices
        self.I_matrix = None  # [K x N] instructor-course compatibility
        self.R_matrix = None  # [N x M] course-room compatibility
        self.A_matrix = None  # [K x T] instructor availability

        # Problem dimensions
        self.N = 0  # number of courses
        self.M = 0  # number of rooms
        self.K = 0  # number of instructors
        self.T = 0  # time horizon

        self.participants = None  # Dictionary course_id -> participants

    def load_instance(self, instance_path: str) -> None:
        """
        Load a complete problem instance from directory containing CSV files.

        Args:
            instance_path: Path to directory containing instance files
        """
        if not os.path.exists(instance_path):
            raise FileNotFoundError(f"Instance path does not exist: {instance_path}")

        # Extract instance name from path
        self.instance_name = os.path.basename(os.path.normpath(instance_path))

        # Load metadata
        self._load_metadata(instance_path)

        # Update metadata with instance name if not present
        if 'name' not in self.metadata:
            self.metadata['name'] = self.instance_name

        # Load basic data
        self._load_basic_data(instance_path)

        # Set problem dimensions
        self.N = len(self.courses)
        self.M = len(self.rooms)
        self.K = len(self.instructors)
        self.T = self.metadata.get('time_horizon', 40)

        # Build matrices
        self._build_instructor_course_compatibility(instance_path)
        self._build_room_course_compatibility(instance_path)
        self._build_instructor_availability()

        self._load_participants(instance_path)

        print(f"Loaded instance: {self.metadata.get('name', 'unknown')}")
        print(f"Dimensions: {self.N} courses, {self.M} rooms, {self.K} instructors, {self.T} time slots")

    def _load_metadata(self, instance_path: str) -> None:
        """Load instance metadata from JSON file."""
        metadata_file = os.path.join(instance_path, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            # Default metadata if file doesn't exist
            self.metadata = {
                "name": os.path.basename(instance_path),
                "time_horizon": 40,
                "description": "No description available"
            }

    def _load_basic_data(self, instance_path: str) -> None:
        """Load courses, rooms, and instructors from CSV files."""
        # Load courses
        courses_file = os.path.join(instance_path, "courses.csv")
        if not os.path.exists(courses_file):
            raise FileNotFoundError(f"Missing courses.csv in {instance_path}")
        self.courses = pd.read_csv(courses_file)

        # Load rooms
        rooms_file = os.path.join(instance_path, "rooms.csv")
        if not os.path.exists(rooms_file):
            raise FileNotFoundError(f"Missing rooms.csv in {instance_path}")
        self.rooms = pd.read_csv(rooms_file)

        # Load instructors
        instructors_file = os.path.join(instance_path, "instructors.csv")
        if not os.path.exists(instructors_file):
            raise FileNotFoundError(f"Missing instructors.csv in {instance_path}")
        self.instructors = pd.read_csv(instructors_file)

        # Validate data integrity
        self._validate_basic_data()

    def _validate_basic_data(self) -> None:
        """Validate that basic data has required columns and valid values."""
        required_course_cols = ['course_id', 'duration', 'price_per_participant', 'max_participants']
        required_room_cols = ['room_id', 'capacity', 'usage_cost_per_hour', 'idle_cost_per_hour']
        required_instructor_cols = ['instructor_id', 'hourly_rate']

        for col in required_course_cols:
            if col not in self.courses.columns:
                raise ValueError(f"Missing required column in courses.csv: {col}")

        for col in required_room_cols:
            if col not in self.rooms.columns:
                raise ValueError(f"Missing required column in rooms.csv: {col}")

        for col in required_instructor_cols:
            if col not in self.instructors.columns:
                raise ValueError(f"Missing required column in instructors.csv: {col}")

        # Check for positive values where required
        if (self.courses['duration'] <= 0).any():
            raise ValueError("Course duration must be positive")
        if (self.courses['price_per_participant'] <= 0).any():
            raise ValueError("Course price must be positive")
        if (self.rooms['capacity'] <= 0).any():
            raise ValueError("Room capacity must be positive")
        if (self.instructors['hourly_rate'] <= 0).any():
            raise ValueError("Instructor hourly rate must be positive")

    def _build_instructor_course_compatibility(self, instance_path: str) -> None:
        """Build I_matrix [K x N] for instructor-course compatibility."""
        self.I_matrix = np.zeros((self.K, self.N), dtype=int)

        # Try to load from compatibility file first
        compat_file = os.path.join(instance_path, "instructor_course_compatibility.csv")

        if os.path.exists(compat_file):
            # Load compressed format
            compat_df = pd.read_csv(compat_file)

            for _, row in compat_df.iterrows():
                instructor_idx = row['instructor_id'] - 1  # Convert to 0-indexed

                if pd.notna(row['compatible_courses']) and row['compatible_courses']:
                    # Parse semicolon-separated course IDs
                    course_ids = [int(x.strip()) for x in str(row['compatible_courses']).split(';')]
                    course_indices = [cid - 1 for cid in course_ids]  # Convert to 0-indexed

                    # Set compatibility
                    for course_idx in course_indices:
                        if 0 <= course_idx < self.N:
                            self.I_matrix[instructor_idx, course_idx] = 1
        else:
            # If no compatibility file, assume all instructors can teach all courses
            print("Warning: No instructor_course_compatibility.csv found. Assuming all compatible.")
            self.I_matrix = np.ones((self.K, self.N), dtype=int)

        # Ensure each instructor can teach at least one course
        for k in range(self.K):
            if self.I_matrix[k, :].sum() == 0:
                print(f"Warning: Instructor {k + 1} has no compatible courses. Making compatible with course 1.")
                self.I_matrix[k, 0] = 1

    def _build_room_course_compatibility(self, instance_path: str) -> None:
        """Build R_matrix [N x M] for course-room compatibility."""
        self.R_matrix = np.zeros((self.N, self.M), dtype=int)

        # Try to load from compatibility file first
        compat_file = os.path.join(instance_path, "room_course_compatibility.csv")

        if os.path.exists(compat_file):
            # Load compressed format
            compat_df = pd.read_csv(compat_file)

            for _, row in compat_df.iterrows():
                room_idx = row['room_id'] - 1  # Convert to 0-indexed

                if pd.notna(row['compatible_courses']) and row['compatible_courses']:
                    # Parse semicolon-separated course IDs
                    course_ids = [int(x.strip()) for x in str(row['compatible_courses']).split(';')]
                    course_indices = [cid - 1 for cid in course_ids]  # Convert to 0-indexed

                    # Set compatibility
                    for course_idx in course_indices:
                        if 0 <= course_idx < self.N:
                            self.R_matrix[course_idx, room_idx] = 1
        else:
            # If no compatibility file, assume all rooms can host all courses
            print("Warning: No room_course_compatibility.csv found. Assuming all compatible.")
            self.R_matrix = np.ones((self.N, self.M), dtype=int)

        # Ensure each course can be held in at least one room
        for i in range(self.N):
            if self.R_matrix[i, :].sum() == 0:
                print(f"Warning: Course {i + 1} has no compatible rooms. Making compatible with room 1.")
                self.R_matrix[i, 0] = 1

    def _build_instructor_availability(self) -> None:
        """Build A_matrix [K x T] for instructor time availability."""
        self.A_matrix = np.zeros((self.K, self.T), dtype=int)

        for idx, instructor in self.instructors.iterrows():
            instructor_idx = instructor['instructor_id'] - 1  # Convert to 0-indexed

            if 'available_slots' in instructor and pd.notna(instructor['available_slots']):
                # Parse semicolon-separated time slots
                time_slots = [int(x.strip()) for x in str(instructor['available_slots']).split(';')]
                slot_indices = [ts - 1 for ts in time_slots]  # Convert to 0-indexed

                # Set availability
                for slot_idx in slot_indices:
                    if 0 <= slot_idx < self.T:
                        self.A_matrix[instructor_idx, slot_idx] = 1
            else:
                # If no availability specified, assume available all time
                print(
                    f"Warning: No availability specified for instructor {instructor_idx + 1}. Assuming always available.")
                self.A_matrix[instructor_idx, :] = 1

        # Ensure each instructor is available at least one time slot
        for k in range(self.K):
            if self.A_matrix[k, :].sum() == 0:
                print(f"Warning: Instructor {k + 1} has no available slots. Making available in slot 1.")
                self.A_matrix[k, 0] = 1

    def get_course_data(self, course_id: int) -> Dict:
        """Get all data for a specific course."""
        course_idx = course_id - 1
        if course_idx < 0 or course_idx >= self.N:
            raise ValueError(f"Invalid course_id: {course_id}")

        course_row = self.courses.iloc[course_idx]
        return {
            'id': int(course_row['course_id']),
            'duration': int(course_row['duration']),
            'price': float(course_row['price_per_participant']),
            'max_participants': int(course_row['max_participants']),
            'compatible_instructors': np.where(self.I_matrix[:, course_idx] == 1)[0] + 1,  # Convert back to 1-indexed
            'compatible_rooms': np.where(self.R_matrix[course_idx, :] == 1)[0] + 1  # Convert back to 1-indexed
        }

    def get_room_data(self, room_id: int) -> Dict:
        """Get all data for a specific room."""
        room_idx = room_id - 1
        if room_idx < 0 or room_idx >= self.M:
            raise ValueError(f"Invalid room_id: {room_id}")

        room_row = self.rooms.iloc[room_idx]
        return {
            'id': int(room_row['room_id']),
            'capacity': int(room_row['capacity']),
            'usage_cost_per_hour': float(room_row['usage_cost_per_hour']),
            'idle_cost_per_hour': float(room_row['idle_cost_per_hour']),
            'compatible_courses': np.where(self.R_matrix[:, room_idx] == 1)[0] + 1  # Convert back to 1-indexed
        }

    def get_instructor_data(self, instructor_id: int) -> Dict:
        """Get all data for a specific instructor."""
        instructor_idx = instructor_id - 1
        if instructor_idx < 0 or instructor_idx >= self.K:
            raise ValueError(f"Invalid instructor_id: {instructor_id}")

        instructor_row = self.instructors.iloc[instructor_idx]
        return {
            'id': int(instructor_row['instructor_id']),
            'hourly_rate': float(instructor_row['hourly_rate']),
            'available_slots': np.where(self.A_matrix[instructor_idx, :] == 1)[0] + 1,  # Convert back to 1-indexed
            'compatible_courses': np.where(self.I_matrix[instructor_idx, :] == 1)[0] + 1  # Convert back to 1-indexed
        }

    def is_instructor_course_compatible(self, instructor_id: int, course_id: int) -> bool:
        """Check if instructor can teach course."""
        return self.I_matrix[instructor_id - 1, course_id - 1] == 1

    def is_room_course_compatible(self, room_id: int, course_id: int) -> bool:
        """Check if room can host course."""
        return self.R_matrix[course_id - 1, room_id - 1] == 1

    def is_instructor_available(self, instructor_id: int, time_slot: int) -> bool:
        """Check if instructor is available at given time slot."""
        return self.A_matrix[instructor_id - 1, time_slot - 1] == 1

    def _load_participants(self, instance_path: str) -> None:
        """Load participant numbers for courses."""
        participants_file = os.path.join(instance_path, "participants.csv")

        if os.path.exists(participants_file):
            # If participants.csv exists, load from there
            participants_df = pd.read_csv(participants_file)
            if 'current_participants' in participants_df.columns and 'course_id' in participants_df.columns:
                self.participants = participants_df.set_index('course_id')['current_participants'].to_dict()
                print(f"Loaded participants from participants.csv: {list(self.participants.values())}")
            else:
                print("Warning: participants.csv exists but missing required columns")
                self.participants = {}
        elif 'current_participants' in self.courses.columns:
            # If column exists in courses.csv
            self.participants = self.courses.set_index('course_id')['current_participants'].to_dict()
            print(f"Loaded participants from courses.csv: {list(self.participants.values())}")
        else:
            # Fallback - use max_participants
            print("Warning: No participant data found. Using max_participants.")
            self.participants = self.courses.set_index('course_id')['max_participants'].to_dict()

    def get_course_participants(self, course_id: int) -> int:
        """Get actual number of participants for a course."""
        if self.participants is None:
            return 0
        return self.participants.get(course_id, 0)

    def print_instance_summary(self) -> None:
        """Print summary of loaded instance."""
        print(f"\n=== Instance Summary ===")
        print(f"Name: {self.metadata.get('name', 'Unknown')}")
        print(f"Description: {self.metadata.get('description', 'No description')}")
        print(f"Courses: {self.N}")
        print(f"Rooms: {self.M}")
        print(f"Instructors: {self.K}")
        print(f"Time horizon: {self.T}")

        print(f"\nInstructor-Course compatibility density: {self.I_matrix.mean():.2%}")
        print(f"Room-Course compatibility density: {self.R_matrix.mean():.2%}")
        print(f"Instructor availability density: {self.A_matrix.mean():.2%}")

        # Show courses with constraints
        print(f"\nCourse constraints:")
        for i in range(self.N):
            course_data = self.get_course_data(i + 1)
            print(f"  Course {i + 1}: duration={course_data['duration']}, "
                  f"price={course_data['price']}, "
                  f"max_participants={course_data['max_participants']}")


# Utility function for easy loading
def load_instance(instance_path: str) -> DataLoader:
    """
    Convenience function to load an instance.

    Args:
        instance_path: Path to instance directory

    Returns:
        DataLoader object with loaded instance
    """
    loader = DataLoader()
    loader.load_instance(instance_path)
    return loader


if __name__ == "__main__":
    # Example usage
    try:
        # Load an instance
        loader = load_instance("data/instances/small/small_01")

        # Print summary
        loader.print_instance_summary()

        # Example queries
        print(f"\nCan instructor 1 teach course 2? {loader.is_instructor_course_compatible(1, 2)}")
        print(f"Can room 1 host course 3? {loader.is_room_course_compatible(1, 3)}")
        print(f"Is instructor 2 available at time 5? {loader.is_instructor_available(2, 5)}")

        # Get detailed data
        course_1_data = loader.get_course_data(1)
        print(f"\nCourse 1 details: {course_1_data}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a valid instance in data/instances/small/small_01/")