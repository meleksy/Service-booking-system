"""
Database Adapter

This module provides a bridge between the SQLite database
and your existing algorithm system. It converts database data
back into DataLoader and CourseSchedulingProblem objects.
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional, Tuple
from database import DatabaseManager

# Import your existing classes
import sys
import os
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.extend([
    os.path.join(project_root, 'src', 'utils'),
    os.path.join(project_root, 'src', 'models')
])

from fix_imports import fix_algorithm_imports
fix_algorithm_imports()

try:
    from data_loader import DataLoader
    from problem import CourseSchedulingProblem
except ImportError as e:
    print(f"Warning: Could not import algorithm classes: {e}")


class DatabaseAdapter:
    """
    Adapts database data to work with existing algorithm system.

    Converts SQLite data back into DataLoader and CourseSchedulingProblem
    objects that your algorithms expect.
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize adapter.

        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager

    def create_data_loader_from_db(self, instance_id: int) -> DataLoader:
        """
        Create DataLoader object from database instance.

        Args:
            instance_id: Database instance ID

        Returns:
            DataLoader object compatible with existing algorithms
        """
        # Get instance info
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT name, description, time_horizon FROM instances WHERE id = ?",
                (instance_id,)
            )
            instance_row = cursor.fetchone()

        if not instance_row:
            raise ValueError(f"Instance {instance_id} not found")

        instance_name = instance_row['name']
        time_horizon = instance_row['time_horizon']

        # Create empty DataLoader
        loader = DataLoader()
        loader.instance_name = instance_name

        # Load basic data
        instructors_data = self.db_manager.get_instructors(instance_id)
        rooms_data = self.db_manager.get_rooms(instance_id)
        courses_data = self.db_manager.get_courses(instance_id)

        # Convert to DataFrames (matching original CSV format)
        loader.instructors = pd.DataFrame([
            {
                'instructor_id': instr['instructor_id'],
                'name': instr['name'],
                'hourly_rate': instr['hourly_rate'],
                'available_slots': instr['available_slots']
            }
            for instr in instructors_data
        ])

        loader.rooms = pd.DataFrame([
            {
                'room_id': room['room_id'],
                'name': room['name'],
                'capacity': room['capacity'],
                'usage_cost_per_hour': room['usage_cost_per_hour'],
                'idle_cost_per_hour': room['idle_cost_per_hour']
            }
            for room in rooms_data
        ])

        loader.courses = pd.DataFrame([
            {
                'course_id': course['course_id'],
                'name': course['name'],
                'duration': course['duration'],
                'price_per_participant': course['price_per_participant'],
                'max_participants': course['max_participants']
            }
            for course in courses_data
        ])

        # Set dimensions
        loader.N = len(loader.courses)
        loader.M = len(loader.rooms)
        loader.K = len(loader.instructors)
        loader.T = time_horizon

        # Build compatibility matrices
        self._build_compatibility_matrices(loader, instance_id)

        # Build instructor availability matrix
        self._build_availability_matrix(loader)

        # Load participant data
        self._load_participant_data(loader, instance_id)

        # Set metadata
        loader.metadata = {
            'name': instance_name,
            'time_horizon': time_horizon,
            'description': f'Loaded from database instance {instance_id}'
        }

        return loader

    def _build_compatibility_matrices(self, loader: DataLoader, instance_id: int) -> None:
        """Build compatibility matrices from database data."""
        # Initialize matrices with zeros
        loader.I_matrix = np.zeros((loader.K, loader.N), dtype=int)
        loader.R_matrix = np.zeros((loader.N, loader.M), dtype=int)

        # Load instructor-course compatibility
        ic_compat = self.db_manager.get_instructor_course_compatibility(instance_id)
        for entry in ic_compat:
            instructor_idx = entry['instructor_id'] - 1  # Convert to 0-indexed
            course_idx = entry['course_id'] - 1          # Convert to 0-indexed
            if entry['is_compatible']:
                loader.I_matrix[instructor_idx, course_idx] = 1

        # Load room-course compatibility
        rc_compat = self.db_manager.get_room_course_compatibility(instance_id)
        for entry in rc_compat:
            course_idx = entry['course_id'] - 1  # Convert to 0-indexed
            room_idx = entry['room_id'] - 1      # Convert to 0-indexed
            if entry['is_compatible']:
                loader.R_matrix[course_idx, room_idx] = 1

    def _build_availability_matrix(self, loader: DataLoader) -> None:
        """Build instructor availability matrix from available_slots data."""
        loader.A_matrix = np.zeros((loader.K, loader.T), dtype=int)

        for idx, instructor in loader.instructors.iterrows():
            instructor_idx = instructor['instructor_id'] - 1  # Convert to 0-indexed
            available_slots = instructor['available_slots']

            if pd.notna(available_slots) and available_slots:
                # Parse semicolon-separated time slots
                try:
                    time_slots = [int(x.strip()) for x in str(available_slots).split(';') if x.strip()]
                    for time_slot in time_slots:
                        slot_idx = time_slot - 1  # Convert to 0-indexed
                        if 0 <= slot_idx < loader.T:
                            loader.A_matrix[instructor_idx, slot_idx] = 1
                except (ValueError, TypeError):
                    # If parsing fails, assume available all time
                    loader.A_matrix[instructor_idx, :] = 1
            else:
                # If no availability specified, assume available all time
                loader.A_matrix[instructor_idx, :] = 1

    def _load_participant_data(self, loader: DataLoader, instance_id: int) -> None:
        """Load participant data from database."""
        loader.participants = {}

        # Get all courses to ensure we have participant data for each
        for idx, course in loader.courses.iterrows():
            course_id = course['course_id']
            # Default to max_participants if no specific data
            loader.participants[course_id] = course['max_participants']

        # Override with actual participant data from database
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT course_id, current_participants FROM participants WHERE instance_id = ?",
                (instance_id,)
            )
            for row in cursor.fetchall():
                loader.participants[row['course_id']] = row['current_participants']

    def create_problem_from_db(self, instance_id: int) -> 'CourseSchedulingProblem':
        """
        Create CourseSchedulingProblem object from database instance.

        Args:
            instance_id: Database instance ID

        Returns:
            CourseSchedulingProblem object ready for algorithms
        """
        # Create DataLoader from database
        loader = self.create_data_loader_from_db(instance_id)

        # Create and return problem
        problem = CourseSchedulingProblem(loader)
        return problem

    def save_algorithm_result(self, instance_id: int, algorithm_name: str,
                             algorithm_variant: str, parameters: Dict,
                             result: Dict, solution_assignments: List[Dict] = None) -> int:
        """
        Save algorithm execution result to database.

        Args:
            instance_id: Database instance ID
            algorithm_name: Name of algorithm used
            algorithm_variant: Variant/configuration name
            parameters: Algorithm parameters used
            result: Result dictionary from algorithm execution
            solution_assignments: List of course assignments

        Returns:
            Schedule run ID
        """
        # Create schedule run record
        run_id = self.db_manager.create_schedule_run(
            instance_id=instance_id,
            algorithm_name=algorithm_name,
            algorithm_variant=algorithm_variant,
            parameters=parameters
        )

        # Update with results
        success = self.db_manager.update_schedule_run(
            run_id=run_id,
            status='completed' if result.get('success') else 'failed',
            execution_time_seconds=result.get('execution_time', 0),
            objective_value=result.get('objective_value'),
            is_feasible=result.get('is_feasible', False),
            num_violations=result.get('num_violations', 0),
            scheduled_courses=result.get('scheduled_courses', 0),
            total_courses=result.get('total_courses', 0),
            room_utilization=result.get('utilization', {}).get('room_utilization'),
            instructor_utilization=result.get('utilization', {}).get('instructor_utilization'),
            courses_scheduled_ratio=result.get('utilization', {}).get('courses_scheduled'),
            error_message=result.get('error_message'),
            detailed_breakdown=result.get('detailed_breakdown')
        )

        # Save individual assignments if provided
        if solution_assignments:
            for assignment in solution_assignments:
                self.db_manager.add_schedule_assignment(
                    run_id=run_id,
                    course_id=assignment['course_id'],
                    room_id=assignment['room_id'],
                    instructor_id=assignment['instructor_id'],
                    start_time=assignment['start_time'],
                    end_time=assignment['end_time'],
                    duration=assignment['duration'],
                    participants=assignment['participants']
                )

        return run_id

    def get_schedule_as_grid(self, run_id: int) -> Dict:
        """
        Get schedule formatted as a grid for display.

        Args:
            run_id: Schedule run ID

        Returns:
            Dictionary with grid data and metadata
        """
        assignments = self.db_manager.get_schedule_assignments(run_id)

        if not assignments:
            return {'grid': {}, 'metadata': {}}

        # Get run info
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute(
                """SELECT sr.*, i.name as instance_name, i.time_horizon
                   FROM schedule_runs sr 
                   JOIN instances i ON sr.instance_id = i.id
                   WHERE sr.id = ?""",
                (run_id,)
            )
            run_info = dict(cursor.fetchone())

        # Get instance data for names
        instance_id = run_info['instance_id']
        courses = self.db_manager.get_courses(instance_id)
        rooms = self.db_manager.get_rooms(instance_id)
        instructors = self.db_manager.get_instructors(instance_id)

        # Create lookup dictionaries
        course_names = {c['course_id']: c['name'] for c in courses}
        room_names = {r['room_id']: r['name'] for r in rooms}
        instructor_names = {i['instructor_id']: i['name'] for i in instructors}

        # Build grid: grid[time_slot][room_id] = assignment_info
        grid = {}
        time_horizon = run_info['time_horizon']

        for assignment in assignments:
            # Safe conversion of IDs
            def safe_int(value):
                if isinstance(value, bytes):
                    return int.from_bytes(value, byteorder='little')
                return int(value)

            start_time = safe_int(assignment['start_time'])
            end_time = safe_int(assignment['end_time'])
            room_id = safe_int(assignment['room_id'])
            course_id = safe_int(assignment['course_id'])
            instructor_id = safe_int(assignment['instructor_id'])
            participants = safe_int(assignment['participants'])

            for t in range(start_time, end_time + 1):
                if t not in grid:
                    grid[t] = {}

                grid[t][room_id] = {
                    'course_id': course_id,
                    'course_name': course_names.get(course_id, f"Course {course_id}"),
                    'instructor_id': instructor_id,
                    'instructor_name': instructor_names.get(instructor_id, f"Instructor {instructor_id}"),
                    'participants': participants,
                    'is_start': t == start_time,
                    'is_end': t == end_time,
                    'duration': assignment['duration']
                }

        # Ensure all time slots exist
        for t in range(1, time_horizon + 1):
            if t not in grid:
                grid[t] = {}

        return {
            'grid': grid,
            'metadata': {
                'run_id': run_id,
                'algorithm_name': run_info['algorithm_name'],
                'algorithm_variant': run_info['algorithm_variant'],
                'instance_name': run_info['instance_name'],
                'time_horizon': time_horizon,
                'objective_value': run_info['objective_value'],
                'is_feasible': run_info['is_feasible'],
                'execution_time': run_info['execution_time_seconds'],
                'scheduled_courses': run_info['scheduled_courses'],
                'total_courses': run_info['total_courses'],
                'room_names': room_names,
                'instructor_names': instructor_names,
                'course_names': course_names
            }
        }

    def export_instance_to_csv(self, instance_id: int, output_dir: str) -> None:
        """
        Export database instance back to CSV format.

        Args:
            instance_id: Database instance ID
            output_dir: Output directory for CSV files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Get instance data
        instructors = self.db_manager.get_instructors(instance_id)
        rooms = self.db_manager.get_rooms(instance_id)
        courses = self.db_manager.get_courses(instance_id)

        # Export instructors
        instructors_df = pd.DataFrame([
            {
                'instructor_id': instr['instructor_id'],
                'name': instr['name'],
                'hourly_rate': instr['hourly_rate'],
                'available_slots': instr['available_slots']
            }
            for instr in instructors
        ])
        instructors_df.to_csv(os.path.join(output_dir, 'instructors.csv'), index=False)

        # Export rooms
        rooms_df = pd.DataFrame([
            {
                'room_id': room['room_id'],
                'name': room['name'],
                'capacity': room['capacity'],
                'usage_cost_per_hour': room['usage_cost_per_hour'],
                'idle_cost_per_hour': room['idle_cost_per_hour']
            }
            for room in rooms
        ])
        rooms_df.to_csv(os.path.join(output_dir, 'rooms.csv'), index=False)

        # Export courses
        courses_df = pd.DataFrame([
            {
                'course_id': course['course_id'],
                'name': course['name'],
                'duration': course['duration'],
                'price_per_participant': course['price_per_participant'],
                'max_participants': course['max_participants']
            }
            for course in courses
        ])
        courses_df.to_csv(os.path.join(output_dir, 'courses.csv'), index=False)

        # Export participants
        participants_data = []
        for course in courses:
            participants_data.append({
                'course_id': course['course_id'],
                'current_participants': course['current_participants']
            })
        participants_df = pd.DataFrame(participants_data)
        participants_df.to_csv(os.path.join(output_dir, 'participants.csv'), index=False)

        # Export compatibility matrices
        ic_compat = self.db_manager.get_instructor_course_compatibility(instance_id)
        rc_compat = self.db_manager.get_room_course_compatibility(instance_id)

        # Instructor-course compatibility (compressed format)
        ic_compressed = {}
        for entry in ic_compat:
            if entry['is_compatible']:
                instructor_id = entry['instructor_id']
                if instructor_id not in ic_compressed:
                    ic_compressed[instructor_id] = []
                ic_compressed[instructor_id].append(entry['course_id'])

        ic_df = pd.DataFrame([
            {
                'instructor_id': instructor_id,
                'compatible_courses': ';'.join(map(str, courses))
            }
            for instructor_id, courses in ic_compressed.items()
        ])
        ic_df.to_csv(os.path.join(output_dir, 'instructor_course_compatibility.csv'), index=False)

        # Room-course compatibility (compressed format)
        rc_compressed = {}
        for entry in rc_compat:
            if entry['is_compatible']:
                room_id = entry['room_id']
                if room_id not in rc_compressed:
                    rc_compressed[room_id] = []
                rc_compressed[room_id].append(entry['course_id'])

        rc_df = pd.DataFrame([
            {
                'room_id': room_id,
                'compatible_courses': ';'.join(map(str, courses))
            }
            for room_id, courses in rc_compressed.items()
        ])
        rc_df.to_csv(os.path.join(output_dir, 'room_course_compatibility.csv'), index=False)

        # Export metadata
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT name, description, time_horizon, difficulty FROM instances WHERE id = ?",
                (instance_id,)
            )
            instance_info = dict(cursor.fetchone())

        metadata = {
            'name': instance_info['name'],
            'description': instance_info['description'],
            'time_horizon': instance_info['time_horizon'],
            'difficulty': instance_info['difficulty'],
            'created_date': '2025-01-01',  # Could add to database schema
            'exported_from_db': True
        }

        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Exported instance to {output_dir}")


# Utility functions for easy access
def load_problem_from_db(db_path: str, instance_id: int) -> 'CourseSchedulingProblem':
    """
    Convenience function to load problem from database.

    Args:
        db_path: Path to database file
        instance_id: Instance ID

    Returns:
        CourseSchedulingProblem ready for algorithms
    """
    db_manager = DatabaseManager(db_path)
    adapter = DatabaseAdapter(db_manager)
    return adapter.create_problem_from_db(instance_id)


def load_problem_by_name(db_path: str, instance_name: str) -> 'CourseSchedulingProblem':
    """
    Convenience function to load problem by instance name.

    Args:
        db_path: Path to database file
        instance_name: Instance name

    Returns:
        CourseSchedulingProblem ready for algorithms
    """
    db_manager = DatabaseManager(db_path)
    instance = db_manager.get_instance_by_name(instance_name)

    if not instance:
        raise ValueError(f"Instance '{instance_name}' not found in database")

    adapter = DatabaseAdapter(db_manager)
    return adapter.create_problem_from_db(instance['id'])


if __name__ == "__main__":
    # Test the adapter
    print("Testing DatabaseAdapter...")

    try:
        # Create database manager
        db_manager = DatabaseManager("test_course_scheduling.db")
        db_manager.initialize_database()

        # Create adapter
        adapter = DatabaseAdapter(db_manager)

        # Test with existing instance (if any)
        instances = db_manager.get_instances()
        if instances:
            instance_id = instances[0]['id']
            print(f"Testing with instance ID: {instance_id}")

            # Test loading DataLoader
            loader = adapter.create_data_loader_from_db(instance_id)
            print(f"Loaded DataLoader: {loader.N} courses, {loader.M} rooms, {loader.K} instructors")

            # Test loading Problem
            problem = adapter.create_problem_from_db(instance_id)
            print(f"Created problem for {problem.instance_name}")

            print("✓ DatabaseAdapter test successful!")
        else:
            print("No instances in database. Run migration first.")

    except Exception as e:
        print(f"✗ DatabaseAdapter test failed: {e}")
        import traceback
        traceback.print_exc()