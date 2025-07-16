import sqlite3
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database operations for the course scheduling system.

    Provides methods for CRUD operations, data migration, and integration
    with the existing algorithm system.
    """

    def __init__(self, db_path: str = "course_scheduling.db"):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.schema_path = Path(__file__).parent / "schema.sql"

    def initialize_database(self) -> None:
        """Initialize database with schema if it doesn't exist."""
        if not os.path.exists(self.db_path):
            logger.info(f"Creating new database: {self.db_path}")
            self.create_schema()
        else:
            logger.info(f"Using existing database: {self.db_path}")

    def create_schema(self) -> None:
        """Create database schema from SQL file."""
        try:
            with open(self.schema_path, 'r') as f:
                schema_sql = f.read()

            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(schema_sql)
                conn.commit()

            logger.info("Database schema created successfully")

        except Exception as e:
            logger.error(f"Error creating schema: {e}")
            raise

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn

    # Instance management
    def create_instance(self, name: str, description: str = "",
                        time_horizon: int = 20, difficulty: str = "medium") -> int:
        """
        Create new problem instance.

        Args:
            name: Instance name
            description: Instance description
            time_horizon: Number of time slots
            difficulty: Instance difficulty level

        Returns:
            Instance ID
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO instances (name, description, time_horizon, difficulty)
                   VALUES (?, ?, ?, ?)""",
                (name, description, time_horizon, difficulty)
            )
            instance_id = cursor.lastrowid
            conn.commit()

        logger.info(f"Created instance '{name}' with ID {instance_id}")
        return instance_id

    def get_instances(self) -> List[Dict]:
        """Get all instances."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """SELECT id, name, description, time_horizon, difficulty, 
                          created_date, is_active
                   FROM instances 
                   ORDER BY created_date DESC"""
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_instance_by_name(self, name: str) -> Optional[Dict]:
        """Get instance by name."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """SELECT id, name, description, time_horizon, difficulty, 
                          created_date, is_active
                   FROM instances 
                   WHERE name = ?""",
                (name,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    # Instructor management
    def add_instructor(self, instance_id: int, instructor_id: int, name: str,
                       hourly_rate: float, available_slots: str = "") -> int:
        """
        Add instructor to instance.

        Args:
            instance_id: Instance ID
            instructor_id: Original instructor ID (1-indexed)
            name: Instructor name
            hourly_rate: Hourly rate
            available_slots: Semicolon-separated time slots

        Returns:
            Database record ID
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO instructors (instance_id, instructor_id, name, hourly_rate, available_slots)
                   VALUES (?, ?, ?, ?, ?)""",
                (instance_id, instructor_id, name, hourly_rate, available_slots)
            )
            record_id = cursor.lastrowid
            conn.commit()

        logger.info(f"Added instructor '{name}' to instance {instance_id}")
        return record_id

    def get_instructors(self, instance_id: int) -> List[Dict]:
        """Get all instructors for instance."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """SELECT id, instructor_id, name, hourly_rate, available_slots, created_date
                   FROM instructors 
                   WHERE instance_id = ? 
                   ORDER BY instructor_id""",
                (instance_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def update_instructor(self, instance_id: int, instructor_id: int,
                          name: str = None, hourly_rate: float = None,
                          available_slots: str = None) -> bool:
        """Update instructor data."""
        updates = []
        params = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if hourly_rate is not None:
            updates.append("hourly_rate = ?")
            params.append(hourly_rate)
        if available_slots is not None:
            updates.append("available_slots = ?")
            params.append(available_slots)

        if not updates:
            return False

        params.extend([instance_id, instructor_id])

        with self.get_connection() as conn:
            cursor = conn.execute(
                f"""UPDATE instructors 
                    SET {', '.join(updates)}
                    WHERE instance_id = ? AND instructor_id = ?""",
                params
            )
            conn.commit()

        return cursor.rowcount > 0

    def delete_instructor(self, instance_id: int, instructor_id: int) -> bool:
        """Delete instructor from instance."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """DELETE FROM instructors 
                   WHERE instance_id = ? AND instructor_id = ?""",
                (instance_id, instructor_id)
            )
            conn.commit()

        return cursor.rowcount > 0

    # Room management
    def add_room(self, instance_id: int, room_id: int, name: str, capacity: int,
                 usage_cost_per_hour: float = 0, idle_cost_per_hour: float = 0) -> int:
        """Add room to instance."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO rooms (instance_id, room_id, name, capacity, 
                                     usage_cost_per_hour, idle_cost_per_hour)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (instance_id, room_id, name, capacity, usage_cost_per_hour, idle_cost_per_hour)
            )
            record_id = cursor.lastrowid
            conn.commit()

        logger.info(f"Added room '{name}' to instance {instance_id}")
        return record_id

    def get_rooms(self, instance_id: int) -> List[Dict]:
        """Get all rooms for instance."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """SELECT id, room_id, name, capacity, usage_cost_per_hour, 
                          idle_cost_per_hour, created_date
                   FROM rooms 
                   WHERE instance_id = ? 
                   ORDER BY room_id""",
                (instance_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def update_room(self, instance_id: int, room_id: int, **kwargs) -> bool:
        """Update room data."""
        allowed_fields = ['name', 'capacity', 'usage_cost_per_hour', 'idle_cost_per_hour']
        updates = []
        params = []

        for field, value in kwargs.items():
            if field in allowed_fields and value is not None:
                updates.append(f"{field} = ?")
                params.append(value)

        if not updates:
            return False

        params.extend([instance_id, room_id])

        with self.get_connection() as conn:
            cursor = conn.execute(
                f"""UPDATE rooms 
                    SET {', '.join(updates)}
                    WHERE instance_id = ? AND room_id = ?""",
                params
            )
            conn.commit()

        return cursor.rowcount > 0

    def delete_room(self, instance_id: int, room_id: int) -> bool:
        """Delete room from instance."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """DELETE FROM rooms 
                   WHERE instance_id = ? AND room_id = ?""",
                (instance_id, room_id)
            )
            conn.commit()

        return cursor.rowcount > 0

    # Course management
    def add_course(self, instance_id: int, course_id: int, name: str, duration: int,
                   price_per_participant: float, max_participants: int) -> int:
        """Add course to instance."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO courses (instance_id, course_id, name, duration, 
                                       price_per_participant, max_participants)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (instance_id, course_id, name, duration, price_per_participant, max_participants)
            )
            record_id = cursor.lastrowid
            conn.commit()

        logger.info(f"Added course '{name}' to instance {instance_id}")
        return record_id

    def get_courses(self, instance_id: int) -> List[Dict]:
        """Get all courses for instance."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """SELECT c.id, c.course_id, c.name, c.duration, 
                          c.price_per_participant, c.max_participants, c.created_date,
                          COALESCE(p.current_participants, 0) as current_participants
                   FROM courses c
                   LEFT JOIN participants p ON c.instance_id = p.instance_id 
                                           AND c.course_id = p.course_id
                   WHERE c.instance_id = ? 
                   ORDER BY c.course_id""",
                (instance_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def update_course(self, instance_id: int, course_id: int, **kwargs) -> bool:
        """Update course data."""
        allowed_fields = ['name', 'duration', 'price_per_participant', 'max_participants']
        updates = []
        params = []

        for field, value in kwargs.items():
            if field in allowed_fields and value is not None:
                updates.append(f"{field} = ?")
                params.append(value)

        if not updates:
            return False

        params.extend([instance_id, course_id])

        with self.get_connection() as conn:
            cursor = conn.execute(
                f"""UPDATE courses 
                    SET {', '.join(updates)}
                    WHERE instance_id = ? AND course_id = ?""",
                params
            )
            conn.commit()

        return cursor.rowcount > 0

    def set_course_participants(self, instance_id: int, course_id: int,
                                current_participants: int) -> None:
        """Set current number of participants for course."""
        with self.get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO participants 
                   (instance_id, course_id, current_participants, updated_date)
                   VALUES (?, ?, ?, ?)""",
                (instance_id, course_id, current_participants, datetime.now().isoformat())
            )
            conn.commit()

    # Compatibility management
    def set_instructor_course_compatibility(self, instance_id: int, instructor_id: int,
                                            course_id: int, is_compatible: bool) -> None:
        """Set instructor-course compatibility."""
        with self.get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO compatibility_instructor_course 
                   (instance_id, instructor_id, course_id, is_compatible)
                   VALUES (?, ?, ?, ?)""",
                (instance_id, instructor_id, course_id, is_compatible)
            )
            conn.commit()

    def set_room_course_compatibility(self, instance_id: int, room_id: int,
                                      course_id: int, is_compatible: bool) -> None:
        """Set room-course compatibility."""
        with self.get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO compatibility_room_course 
                   (instance_id, room_id, course_id, is_compatible)
                   VALUES (?, ?, ?, ?)""",
                (instance_id, room_id, course_id, is_compatible)
            )
            conn.commit()

    def get_instructor_course_compatibility(self, instance_id: int) -> List[Dict]:
        """Get all instructor-course compatibility entries."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """SELECT cic.instructor_id, cic.course_id, cic.is_compatible,
                          i.name as instructor_name, c.name as course_name
                   FROM compatibility_instructor_course cic
                   JOIN instructors i ON cic.instance_id = i.instance_id 
                                     AND cic.instructor_id = i.instructor_id
                   JOIN courses c ON cic.instance_id = c.instance_id 
                                 AND cic.course_id = c.course_id
                   WHERE cic.instance_id = ?
                   ORDER BY cic.instructor_id, cic.course_id""",
                (instance_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_room_course_compatibility(self, instance_id: int) -> List[Dict]:
        """Get all room-course compatibility entries."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """SELECT crc.room_id, crc.course_id, crc.is_compatible,
                          r.name as room_name, c.name as course_name
                   FROM compatibility_room_course crc
                   JOIN rooms r ON crc.instance_id = r.instance_id 
                               AND crc.room_id = r.room_id
                   JOIN courses c ON crc.instance_id = c.instance_id 
                                 AND crc.course_id = c.course_id
                   WHERE crc.instance_id = ?
                   ORDER BY crc.room_id, crc.course_id""",
                (instance_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    # Algorithm run management
    def create_schedule_run(self, instance_id: int, algorithm_name: str,
                            algorithm_variant: str = None, parameters: Dict = None) -> int:
        """Create new algorithm run record."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO schedule_runs 
                   (instance_id, algorithm_name, algorithm_variant, parameters, status)
                   VALUES (?, ?, ?, ?, 'pending')""",
                (instance_id, algorithm_name, algorithm_variant,
                 json.dumps(parameters) if parameters else None)
            )
            run_id = cursor.lastrowid
            conn.commit()

        logger.info(f"Created schedule run {run_id} for instance {instance_id}")
        return run_id

    def update_schedule_run(self, run_id: int, status: str = None,
                            execution_time_seconds: float = None,
                            objective_value: float = None, is_feasible: bool = None,
                            num_violations: int = None, scheduled_courses: int = None,
                            total_courses: int = None, room_utilization: float = None,
                            instructor_utilization: float = None,
                            courses_scheduled_ratio: float = None,
                            error_message: str = None,
                            detailed_breakdown: Dict = None) -> bool:
        """Update algorithm run results."""
        updates = []
        params = []

        fields = {
            'status': status,
            'execution_time_seconds': execution_time_seconds,
            'objective_value': objective_value,
            'is_feasible': is_feasible,
            'num_violations': num_violations,
            'scheduled_courses': scheduled_courses,
            'total_courses': total_courses,
            'room_utilization': room_utilization,
            'instructor_utilization': instructor_utilization,
            'courses_scheduled_ratio': courses_scheduled_ratio,
            'error_message': error_message
        }

        for field, value in fields.items():
            if value is not None:
                updates.append(f"{field} = ?")
                params.append(value)

        if detailed_breakdown is not None:
            updates.append("detailed_breakdown = ?")
            params.append(json.dumps(detailed_breakdown))

        if status in ['completed', 'failed']:
            updates.append("end_time = ?")
            params.append(datetime.now().isoformat())

        if not updates:
            return False

        params.append(run_id)

        with self.get_connection() as conn:
            cursor = conn.execute(
                f"""UPDATE schedule_runs 
                    SET {', '.join(updates)}
                    WHERE id = ?""",
                params
            )
            conn.commit()

        return cursor.rowcount > 0

    def add_schedule_assignment(self, run_id: int, course_id: int, room_id: int,
                                instructor_id: int, start_time: int, end_time: int,
                                duration: int, participants: int) -> int:
        """Add assignment to schedule run."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO schedule_assignments 
                   (run_id, course_id, room_id, instructor_id, start_time, 
                    end_time, duration, participants)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, course_id, room_id, instructor_id, start_time,
                 end_time, duration, participants)
            )
            assignment_id = cursor.lastrowid
            conn.commit()

        return assignment_id

    def get_schedule_runs(self, instance_id: int = None, limit: int = 50) -> List[Dict]:
        """Get algorithm runs with summary information."""
        query = """SELECT * FROM v_schedule_summary"""
        params = []

        if instance_id is not None:
            query += " WHERE instance_id = ?"
            params.append(instance_id)

        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)

        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_schedule_assignments(self, run_id: int) -> List[Dict]:
        """Get assignments for specific run."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """SELECT course_id, room_id, instructor_id, start_time, 
                          end_time, duration, participants
                   FROM schedule_assignments 
                   WHERE run_id = ?
                   ORDER BY start_time, room_id""",
                (run_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_schedule_run(self, run_id: int) -> bool:
        """Delete schedule run and all its assignments."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """DELETE FROM schedule_runs WHERE id = ?""",
                (run_id,)
            )
            conn.commit()

        return cursor.rowcount > 0

        # Client management
    def add_client(self, first_name: str, last_name: str, email: str = None,
                   phone: str = None) -> int:
        """Add new client."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO clients (first_name, last_name, email, phone)
                   VALUES (?, ?, ?, ?)""",
                (first_name, last_name, email, phone)
            )
            client_id = cursor.lastrowid
            conn.commit()

        logger.info(f"Added client '{first_name} {last_name}' with ID {client_id}")
        return client_id

    def get_clients(self, active_only: bool = True) -> List[Dict]:
        """Get all clients."""
        query = """SELECT id, first_name, last_name, email, phone, created_date, is_active
                   FROM clients"""
        params = []

        if active_only:
            query += " WHERE is_active = 1"

        query += " ORDER BY id"  # Zmienione z "last_name, first_name" na "id"

        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def update_client(self, client_id: int, **kwargs) -> bool:
        """Update client data."""
        allowed_fields = ['first_name', 'last_name', 'email', 'phone', 'is_active']
        updates = []
        params = []

        for field, value in kwargs.items():
            if field in allowed_fields and value is not None:
                updates.append(f"{field} = ?")
                params.append(value)

        if not updates:
            return False

        params.append(client_id)

        with self.get_connection() as conn:
            cursor = conn.execute(
                f"""UPDATE clients 
                    SET {', '.join(updates)}
                    WHERE id = ?""",
                params
            )
            conn.commit()

        return cursor.rowcount > 0

    # Reservation management
    def add_reservation(self, instance_id: int, client_id: int, course_id: int,
                        status: str = 'pending', notes: str = None) -> int:
        """Add new reservation."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO reservations (instance_id, client_id, course_id, status, notes)
                   VALUES (?, ?, ?, ?, ?)""",
                (instance_id, client_id, course_id, status, notes)
            )
            reservation_id = cursor.lastrowid
            conn.commit()

        logger.info(f"Added reservation {reservation_id} for client {client_id}")
        return reservation_id

    def get_reservations(self, instance_id: int = None, status: str = None) -> List[Dict]:
        """Get reservations with full details."""
        query = "SELECT * FROM v_reservations_full"
        params = []
        conditions = []

        if instance_id is not None:
            conditions.append("instance_id = ?")
            params.append(instance_id)

        if status is not None:
            conditions.append("status = ?")
            params.append(status)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_date DESC"

        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def update_reservation(self, reservation_id: int, **kwargs) -> bool:
        """Update reservation data."""
        allowed_fields = ['status', 'notes', 'updated_date']
        updates = []
        params = []

        for field, value in kwargs.items():
            if field in allowed_fields and value is not None:
                updates.append(f"{field} = ?")
                params.append(value)

        # Always update the updated_date
        if 'updated_date' not in kwargs:
            updates.append("updated_date = ?")
            params.append(datetime.now().isoformat())

        if not updates:
            return False

        params.append(reservation_id)

        with self.get_connection() as conn:
            cursor = conn.execute(
                f"""UPDATE reservations 
                    SET {', '.join(updates)}
                    WHERE id = ?""",
                params
            )
            conn.commit()

        return cursor.rowcount > 0

    def delete_reservation(self, reservation_id: int) -> bool:
        """Delete reservation."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """DELETE FROM reservations WHERE id = ?""",
                (reservation_id,)
            )
            conn.commit()

        return cursor.rowcount > 0

    def get_reservation_stats(self, instance_id: int) -> Dict:
        """Get reservation statistics for instance."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """SELECT 
                       COUNT(*) as total_reservations,
                       SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                       SUM(CASE WHEN status = 'confirmed' THEN 1 ELSE 0 END) as confirmed,
                       SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled
                   FROM reservations 
                   WHERE instance_id = ?""",
                (instance_id,)
            )
            return dict(cursor.fetchone())