"""
CSV to Database Migration Script

This script migrates data from your existing CSV-based instances
to the SQLite database format.

Usage:
    python migrate_csv.py --instance_path "data/instances/small/cplex_test" --db_path "course_scheduling.db"
    python migrate_csv.py --migrate_all --instances_dir "data/instances" --db_path "course_scheduling.db"
"""

import os
import sys
import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add your existing utils to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src" / "utils"))

from database import DatabaseManager

try:
    from data_loader import load_instance, DataLoader
except ImportError:
    print("Error: Could not import data_loader. Make sure the path is correct.")
    sys.exit(1)


class CSVMigrator:
    """
    Migrates CSV-based course scheduling instances to SQLite database.
    """

    def __init__(self, db_path: str = "course_scheduling.db"):
        """
        Initialize migrator.

        Args:
            db_path: Path to SQLite database
        """
        self.db_manager = DatabaseManager(db_path)
        self.db_manager.initialize_database()

    def migrate_instance(self, instance_path: str, force: bool = False) -> Optional[int]:
        """
        Migrate single instance from CSV files to database.

        Args:
            instance_path: Path to instance directory containing CSV files
            force: Whether to overwrite existing instance

        Returns:
            Instance ID if successful, None otherwise
        """
        instance_path = Path(instance_path)
        instance_name = instance_path.name

        print(f"Migrating instance: {instance_name}")
        print(f"Source path: {instance_path}")

        # Check if instance already exists
        existing = self.db_manager.get_instance_by_name(instance_name)
        if existing and not force:
            print(f"Instance '{instance_name}' already exists. Use --force to overwrite.")
            return existing['id']

        try:
            # Load instance using your existing data loader
            loader = load_instance(str(instance_path))

            # Load metadata
            metadata_path = instance_path / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

            # Create or update instance
            if existing:
                instance_id = existing['id']
                print(f"Overwriting existing instance ID: {instance_id}")
                # Delete existing data (will cascade)
                with self.db_manager.get_connection() as conn:
                    conn.execute("DELETE FROM instances WHERE id = ?", (instance_id,))
                    conn.commit()

            # Create new instance
            instance_id = self.db_manager.create_instance(
                name=instance_name,
                description=metadata.get('description', f'Migrated from {instance_path}'),
                time_horizon=metadata.get('time_horizon', loader.T),
                difficulty=metadata.get('difficulty', 'medium')
            )

            # Migrate instructors
            self._migrate_instructors(loader, instance_id)

            # Migrate rooms
            self._migrate_rooms(loader, instance_id)

            # Migrate courses
            self._migrate_courses(loader, instance_id)

            # Migrate participants
            self._migrate_participants(loader, instance_id)

            # Migrate compatibility matrices
            self._migrate_compatibility(loader, instance_id)

            print(f"✓ Successfully migrated instance '{instance_name}' with ID {instance_id}")
            return instance_id

        except Exception as e:
            print(f"✗ Error migrating instance '{instance_name}': {e}")
            import traceback
            traceback.print_exc()
            return None

    def _migrate_instructors(self, loader: DataLoader, instance_id: int) -> None:
        """Migrate instructors from CSV to database."""
        print("  Migrating instructors...")

        for idx, instructor in loader.instructors.iterrows():
            instructor_id = int(instructor['instructor_id'])
            name = instructor['name']
            hourly_rate = float(instructor['hourly_rate'])

            # Convert available slots
            available_slots = ""
            if 'available_slots' in instructor and pd.notna(instructor['available_slots']):
                available_slots = str(instructor['available_slots']).strip()

            self.db_manager.add_instructor(
                instance_id=instance_id,
                instructor_id=instructor_id,
                name=name,
                hourly_rate=hourly_rate,
                available_slots=available_slots
            )

        print(f"    Added {len(loader.instructors)} instructors")

    def _migrate_rooms(self, loader: DataLoader, instance_id: int) -> None:
        """Migrate rooms from CSV to database."""
        print("  Migrating rooms...")

        for idx, room in loader.rooms.iterrows():
            room_id = int(room['room_id'])
            name = room['name']
            capacity = int(room['capacity'])
            usage_cost = float(room.get('usage_cost_per_hour', 0))
            idle_cost = float(room.get('idle_cost_per_hour', 0))

            self.db_manager.add_room(
                instance_id=instance_id,
                room_id=room_id,
                name=name,
                capacity=capacity,
                usage_cost_per_hour=usage_cost,
                idle_cost_per_hour=idle_cost
            )

        print(f"    Added {len(loader.rooms)} rooms")

    def _migrate_courses(self, loader: DataLoader, instance_id: int) -> None:
        """Migrate courses from CSV to database."""
        print("  Migrating courses...")

        for idx, course in loader.courses.iterrows():
            course_id = int(course['course_id'])
            name = course['name']
            duration = int(course['duration'])
            price_per_participant = float(course['price_per_participant'])
            max_participants = int(course['max_participants'])

            self.db_manager.add_course(
                instance_id=instance_id,
                course_id=course_id,
                name=name,
                duration=duration,
                price_per_participant=price_per_participant,
                max_participants=max_participants
            )

        print(f"    Added {len(loader.courses)} courses")

    def _migrate_participants(self, loader: DataLoader, instance_id: int) -> None:
        """Migrate participant numbers from CSV to database."""
        print("  Migrating participants...")

        if hasattr(loader, 'participants') and loader.participants:
            count = 0
            for course_id, participants in loader.participants.items():
                self.db_manager.set_course_participants(
                    instance_id=instance_id,
                    course_id=course_id,
                    current_participants=participants
                )
                count += 1
            print(f"    Added participant data for {count} courses")
        else:
            # If no participants data, use max_participants as default
            for idx, course in loader.courses.iterrows():
                course_id = int(course['course_id'])
                max_participants = int(course['max_participants'])

                self.db_manager.set_course_participants(
                    instance_id=instance_id,
                    course_id=course_id,
                    current_participants=max_participants
                )

            print(f"    Set default participants for {len(loader.courses)} courses")

    def _migrate_compatibility(self, loader: DataLoader, instance_id: int) -> None:
        """Migrate compatibility matrices from CSV to database."""
        print("  Migrating compatibility matrices...")

        # Instructor-Course compatibility
        instructor_course_count = 0
        for k in range(loader.K):  # instructors
            for i in range(loader.N):  # courses
                instructor_id = k + 1  # Convert to 1-indexed
                course_id = i + 1  # Convert to 1-indexed
                is_compatible = bool(loader.I_matrix[k, i])

                self.db_manager.set_instructor_course_compatibility(
                    instance_id=instance_id,
                    instructor_id=instructor_id,
                    course_id=course_id,
                    is_compatible=is_compatible
                )
                instructor_course_count += 1

        # Room-Course compatibility
        room_course_count = 0
        for i in range(loader.N):  # courses
            for m in range(loader.M):  # rooms
                course_id = i + 1  # Convert to 1-indexed
                room_id = m + 1  # Convert to 1-indexed
                is_compatible = bool(loader.R_matrix[i, m])

                self.db_manager.set_room_course_compatibility(
                    instance_id=instance_id,
                    room_id=room_id,
                    course_id=course_id,
                    is_compatible=is_compatible
                )
                room_course_count += 1

        print(f"    Added {instructor_course_count} instructor-course compatibility entries")
        print(f"    Added {room_course_count} room-course compatibility entries")

    def migrate_all_instances(self, instances_dir: str, force: bool = False) -> Dict[str, Optional[int]]:
        """
        Migrate all instances from directory structure.

        Args:
            instances_dir: Root directory containing instance categories
            force: Whether to overwrite existing instances

        Returns:
            Dictionary mapping instance names to their database IDs
        """
        instances_dir = Path(instances_dir)
        results = {}

        print(f"Scanning for instances in: {instances_dir}")

        # Scan directory structure: instances_dir/category/instance_name/
        for category_dir in instances_dir.iterdir():
            if not category_dir.is_dir():
                continue

            print(f"\nCategory: {category_dir.name}")

            for instance_dir in category_dir.iterdir():
                if not instance_dir.is_dir():
                    continue

                # Check if this looks like an instance (has required CSV files)
                required_files = ['courses.csv', 'rooms.csv', 'instructors.csv']
                if not all((instance_dir / file).exists() for file in required_files):
                    print(f"  Skipping {instance_dir.name} (missing required CSV files)")
                    continue

                instance_name = instance_dir.name
                instance_id = self.migrate_instance(str(instance_dir), force=force)
                results[instance_name] = instance_id

        return results

    def verify_migration(self, instance_name: str) -> bool:
        """
        Verify that migration was successful by comparing data.

        Args:
            instance_name: Name of migrated instance

        Returns:
            True if verification passed
        """
        print(f"\nVerifying migration for instance: {instance_name}")

        # Get instance from database
        instance = self.db_manager.get_instance_by_name(instance_name)
        if not instance:
            print("✗ Instance not found in database")
            return False

        instance_id = instance['id']

        # Check data counts
        instructors = self.db_manager.get_instructors(instance_id)
        rooms = self.db_manager.get_rooms(instance_id)
        courses = self.db_manager.get_courses(instance_id)

        print(f"  Instructors: {len(instructors)}")
        print(f"  Rooms: {len(rooms)}")
        print(f"  Courses: {len(courses)}")

        # Basic validation
        if len(instructors) == 0 or len(rooms) == 0 or len(courses) == 0:
            print("✗ Missing basic data")
            return False

        # Check compatibility data
        ic_compat = self.db_manager.get_instructor_course_compatibility(instance_id)
        rc_compat = self.db_manager.get_room_course_compatibility(instance_id)

        expected_ic = len(instructors) * len(courses)
        expected_rc = len(rooms) * len(courses)

        print(f"  Instructor-Course compatibility: {len(ic_compat)}/{expected_ic}")
        print(f"  Room-Course compatibility: {len(rc_compat)}/{expected_rc}")

        if len(ic_compat) != expected_ic or len(rc_compat) != expected_rc:
            print("✗ Compatibility matrix size mismatch")
            return False

        print("✓ Migration verification passed")
        return True


def main():
    """Main migration script."""
    parser = argparse.ArgumentParser(description='Migrate CSV instances to SQLite database')
    parser.add_argument('--db_path', default='course_scheduling.db',
                        help='Path to SQLite database file')
    parser.add_argument('--instance_path', help='Path to single instance directory')
    parser.add_argument('--instances_dir', help='Path to instances root directory')
    parser.add_argument('--migrate_all', action='store_true',
                        help='Migrate all instances from instances_dir')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing instances')
    parser.add_argument('--verify', action='store_true',
                        help='Verify migration after completion')

    args = parser.parse_args()

    if not args.instance_path and not args.migrate_all:
        print("Error: Specify either --instance_path or --migrate_all with --instances_dir")
        return

    migrator = CSVMigrator(args.db_path)

    if args.migrate_all:
        if not args.instances_dir:
            print("Error: --instances_dir required with --migrate_all")
            return

        print(f"Migrating all instances from: {args.instances_dir}")
        print(f"Database: {args.db_path}")
        print("-" * 60)

        results = migrator.migrate_all_instances(args.instances_dir, force=args.force)

        print("\n" + "=" * 60)
        print("MIGRATION SUMMARY")
        print("=" * 60)

        successful = 0
        failed = 0

        for instance_name, instance_id in results.items():
            if instance_id is not None:
                print(f"✓ {instance_name} -> ID {instance_id}")
                successful += 1
            else:
                print(f"✗ {instance_name} -> FAILED")
                failed += 1

        print(f"\nTotal: {successful + failed}, Successful: {successful}, Failed: {failed}")

        # Verification
        if args.verify:
            print("\n" + "=" * 60)
            print("VERIFICATION")
            print("=" * 60)

            for instance_name, instance_id in results.items():
                if instance_id is not None:
                    migrator.verify_migration(instance_name)

    else:
        # Single instance migration
        print(f"Migrating single instance: {args.instance_path}")
        print(f"Database: {args.db_path}")
        print("-" * 60)

        instance_id = migrator.migrate_instance(args.instance_path, force=args.force)

        if instance_id:
            print(f"\n✓ Migration successful! Instance ID: {instance_id}")

            if args.verify:
                instance_name = Path(args.instance_path).name
                migrator.verify_migration(instance_name)
        else:
            print("\n✗ Migration failed!")


if __name__ == "__main__":
    main()