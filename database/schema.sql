-- Course Scheduling Database Schema
-- SQLite database for storing dance school scheduling data

-- Drop tables if they exist (for migrations)
DROP TABLE IF EXISTS schedule_assignments;
DROP TABLE IF EXISTS schedule_runs;
DROP TABLE IF EXISTS compatibility_room_course;
DROP TABLE IF EXISTS compatibility_instructor_course;
DROP TABLE IF EXISTS participants;
DROP TABLE IF EXISTS courses;
DROP TABLE IF EXISTS rooms;
DROP TABLE IF EXISTS instructors;
DROP TABLE IF EXISTS instances;

-- Main data tables
CREATE TABLE instances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    time_horizon INTEGER NOT NULL DEFAULT 20,
    created_date TEXT DEFAULT CURRENT_TIMESTAMP,
    difficulty TEXT DEFAULT 'medium',
    is_active BOOLEAN DEFAULT 1
);

CREATE TABLE instructors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instance_id INTEGER NOT NULL,
    instructor_id INTEGER NOT NULL, -- Original 1-indexed ID from CSV
    name TEXT NOT NULL,
    hourly_rate REAL NOT NULL CHECK (hourly_rate > 0),
    available_slots TEXT, -- Semicolon-separated time slots (e.g., "1;2;3;5;8")
    created_date TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (instance_id) REFERENCES instances(id) ON DELETE CASCADE,
    UNIQUE(instance_id, instructor_id)
);

CREATE TABLE rooms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instance_id INTEGER NOT NULL,
    room_id INTEGER NOT NULL, -- Original 1-indexed ID from CSV
    name TEXT NOT NULL,
    capacity INTEGER NOT NULL CHECK (capacity > 0),
    usage_cost_per_hour REAL NOT NULL DEFAULT 0,
    idle_cost_per_hour REAL NOT NULL DEFAULT 0,
    created_date TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (instance_id) REFERENCES instances(id) ON DELETE CASCADE,
    UNIQUE(instance_id, room_id)
);

CREATE TABLE courses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instance_id INTEGER NOT NULL,
    course_id INTEGER NOT NULL, -- Original 1-indexed ID from CSV
    name TEXT NOT NULL,
    duration INTEGER NOT NULL CHECK (duration > 0),
    price_per_participant REAL NOT NULL CHECK (price_per_participant > 0),
    max_participants INTEGER NOT NULL CHECK (max_participants > 0),
    created_date TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (instance_id) REFERENCES instances(id) ON DELETE CASCADE,
    UNIQUE(instance_id, course_id)
);

CREATE TABLE participants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instance_id INTEGER NOT NULL,
    course_id INTEGER NOT NULL,
    current_participants INTEGER NOT NULL DEFAULT 0 CHECK (current_participants >= 0),
    updated_date TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (instance_id) REFERENCES instances(id) ON DELETE CASCADE,
    FOREIGN KEY (instance_id, course_id) REFERENCES courses(instance_id, course_id) ON DELETE CASCADE,
    UNIQUE(instance_id, course_id)
);

-- Compatibility tables
CREATE TABLE compatibility_instructor_course (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instance_id INTEGER NOT NULL,
    instructor_id INTEGER NOT NULL,
    course_id INTEGER NOT NULL,
    is_compatible BOOLEAN NOT NULL DEFAULT 1,
    created_date TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (instance_id) REFERENCES instances(id) ON DELETE CASCADE,
    FOREIGN KEY (instance_id, instructor_id) REFERENCES instructors(instance_id, instructor_id) ON DELETE CASCADE,
    FOREIGN KEY (instance_id, course_id) REFERENCES courses(instance_id, course_id) ON DELETE CASCADE,
    UNIQUE(instance_id, instructor_id, course_id)
);

CREATE TABLE compatibility_room_course (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instance_id INTEGER NOT NULL,
    room_id INTEGER NOT NULL,
    course_id INTEGER NOT NULL,
    is_compatible BOOLEAN NOT NULL DEFAULT 1,
    created_date TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (instance_id) REFERENCES instances(id) ON DELETE CASCADE,
    FOREIGN KEY (instance_id, room_id) REFERENCES rooms(instance_id, room_id) ON DELETE CASCADE,
    FOREIGN KEY (instance_id, course_id) REFERENCES courses(instance_id, course_id) ON DELETE CASCADE,
    UNIQUE(instance_id, room_id, course_id)
);

-- Algorithm execution tables
CREATE TABLE schedule_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instance_id INTEGER NOT NULL,
    algorithm_name TEXT NOT NULL,
    algorithm_variant TEXT,
    parameters TEXT, -- JSON string with algorithm parameters
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    start_time TEXT DEFAULT CURRENT_TIMESTAMP,
    end_time TEXT,
    execution_time_seconds REAL,
    objective_value REAL,
    is_feasible BOOLEAN,
    num_violations INTEGER DEFAULT 0,
    scheduled_courses INTEGER DEFAULT 0,
    total_courses INTEGER DEFAULT 0,
    room_utilization REAL,
    instructor_utilization REAL,
    courses_scheduled_ratio REAL,
    error_message TEXT,
    detailed_breakdown TEXT, -- JSON string with cost breakdown
    created_date TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (instance_id) REFERENCES instances(id) ON DELETE CASCADE
);

CREATE TABLE schedule_assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    course_id INTEGER NOT NULL,
    room_id INTEGER NOT NULL,
    instructor_id INTEGER NOT NULL,
    start_time INTEGER NOT NULL CHECK (start_time > 0),
    end_time INTEGER NOT NULL CHECK (end_time >= start_time),
    duration INTEGER NOT NULL CHECK (duration > 0),
    participants INTEGER NOT NULL CHECK (participants > 0),
    created_date TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES schedule_runs(id) ON DELETE CASCADE,
    -- Note: We don't add FK to courses/rooms/instructors because
    -- the assignment might reference data from the time of algorithm run
    UNIQUE(run_id, course_id)
);

-- Indexes for better performance
CREATE INDEX idx_instructors_instance ON instructors(instance_id);
CREATE INDEX idx_rooms_instance ON rooms(instance_id);
CREATE INDEX idx_courses_instance ON courses(instance_id);
CREATE INDEX idx_participants_instance ON participants(instance_id);
CREATE INDEX idx_compatibility_ic_instance ON compatibility_instructor_course(instance_id);
CREATE INDEX idx_compatibility_rc_instance ON compatibility_room_course(instance_id);
CREATE INDEX idx_schedule_runs_instance ON schedule_runs(instance_id);
CREATE INDEX idx_schedule_runs_status ON schedule_runs(status);
CREATE INDEX idx_schedule_runs_algorithm ON schedule_runs(algorithm_name);
CREATE INDEX idx_schedule_assignments_run ON schedule_assignments(run_id);
CREATE INDEX idx_schedule_assignments_course ON schedule_assignments(course_id);

-- Views for easier data access
CREATE VIEW v_instructor_availability AS
SELECT
    i.instance_id,
    i.instructor_id,
    i.name as instructor_name,
    i.hourly_rate,
    i.available_slots,
    -- Parse available slots into separate time slots
    GROUP_CONCAT(
        CASE
            WHEN TRIM(value) != '' THEN CAST(TRIM(value) AS INTEGER)
            ELSE NULL
        END
    ) as parsed_slots
FROM instructors i
CROSS JOIN (
    -- Split available_slots by semicolon
    SELECT DISTINCT
        TRIM(
            substr(
                ','||i2.available_slots||',',
                instr(','||i2.available_slots||',', ',')+1,
                instr(','||i2.available_slots||',', ',', instr(','||i2.available_slots||',', ',')+1) - instr(','||i2.available_slots||',', ',')-1
            ),
            ';'
        ) as value
    FROM instructors i2
    WHERE i2.id = i.id
)
GROUP BY i.id, i.instance_id, i.instructor_id, i.name, i.hourly_rate, i.available_slots;

CREATE VIEW v_schedule_summary AS
SELECT
    sr.id as run_id,
    sr.instance_id,
    i.name as instance_name,
    sr.algorithm_name,
    sr.algorithm_variant,
    sr.status,
    sr.start_time,
    sr.execution_time_seconds,
    sr.objective_value,
    sr.is_feasible,
    sr.scheduled_courses,
    sr.total_courses,
    ROUND(sr.room_utilization * 100, 2) as room_utilization_percent,
    ROUND(sr.instructor_utilization * 100, 2) as instructor_utilization_percent,
    ROUND(sr.courses_scheduled_ratio * 100, 2) as courses_scheduled_percent,
    COUNT(sa.id) as assignment_count
FROM schedule_runs sr
LEFT JOIN instances i ON sr.instance_id = i.id
LEFT JOIN schedule_assignments sa ON sr.id = sa.run_id
GROUP BY sr.id, sr.instance_id, i.name, sr.algorithm_name, sr.algorithm_variant,
         sr.status, sr.start_time, sr.execution_time_seconds, sr.objective_value,
         sr.is_feasible, sr.scheduled_courses, sr.total_courses,
         sr.room_utilization, sr.instructor_utilization, sr.courses_scheduled_ratio;

-- Tabela klientów
CREATE TABLE clients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT UNIQUE,
    phone TEXT,
    created_date TEXT DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
);

-- Tabela rezerwacji
CREATE TABLE reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instance_id INTEGER NOT NULL,
    client_id INTEGER NOT NULL,
    course_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'confirmed', 'cancelled')),
    reservation_date TEXT DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    created_date TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_date TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (instance_id) REFERENCES instances(id) ON DELETE CASCADE,
    FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE CASCADE,
    FOREIGN KEY (instance_id, course_id) REFERENCES courses(instance_id, course_id) ON DELETE CASCADE
);

-- Indeksy dla rezerwacji
CREATE INDEX idx_reservations_instance ON reservations(instance_id);
CREATE INDEX idx_reservations_client ON reservations(client_id);
CREATE INDEX idx_reservations_status ON reservations(status);
CREATE INDEX idx_clients_email ON clients(email);

-- Widok z pełnymi informacjami o rezerwacjach
CREATE VIEW v_reservations_full AS
SELECT
    r.id as reservation_id,
    r.instance_id,
    i.name as instance_name,
    r.client_id,
    c.first_name || ' ' || c.last_name as client_name,
    c.email as client_email,
    c.phone as client_phone,
    r.course_id,
    co.name as course_name,
    co.duration as course_duration,
    co.price_per_participant as course_price,
    r.status,
    r.reservation_date,
    r.notes,
    r.created_date,
    r.updated_date
FROM reservations r
LEFT JOIN instances i ON r.instance_id = i.id
LEFT JOIN clients c ON r.client_id = c.id
LEFT JOIN courses co ON r.instance_id = co.instance_id AND r.course_id = co.course_id;


-- Sample data insertion (for testing)
INSERT INTO instances (name, description, time_horizon, difficulty) VALUES
('cplex_test', 'Instance matching CPLEX test data', 5, 'easy'),
('small_01', 'Small test instance', 20, 'medium'),
('demo_instance', 'Demo instance for GUI testing', 15, 'easy');

-- Sample clients
INSERT INTO clients (first_name, last_name, email, phone) VALUES
('Anna', 'Kowalska', 'anna.kowalska@email.com', '+48 123 456 789'),
('Piotr', 'Nowak', 'piotr.nowak@email.com', '+48 987 654 321'),
('Maria', 'Wiśniewska', 'maria.wisniewska@email.com', '+48 555 123 456'),
('Tomasz', 'Kaczmarek', 'tomasz.kaczmarek@email.com', '+48 777 888 999'),
('Katarzyna', 'Zielińska', 'katarzyna.zielinska@email.com', '+48 111 222 333');
