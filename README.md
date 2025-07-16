# Service-booking-system

## About The Project
This repository contains an implementation of various optimization algorithms for solving the Course Scheduling Problem. The system includes multiple algorithms, database management, and a GUI interface for easy interaction with the optimization system.

### Features
- Implementation of multiple optimization algorithms:
  - Greedy Algorithm (with variants)
  - Genetic Algorithm
  - Simulated Annealing
  - Tabu Search
  - Variable Neighborhood Search
- SQLite database for instance and results management
- Tkinter GUI for user-friendly interaction
- CSV data import/export capabilities
- Web-based schedule visualization
- Comprehensive performance analysis tools

## Getting Started

### Prerequisites
- Python 3.8+
- SQLite3 (included with Python)
- 4GB RAM minimum
- Any operating system (Windows/Linux/MacOS)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/meleksy/Service-booking-system
cd course-scheduling-system
```

2. Create and activate virtual environment (recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Project Structure
```
course-scheduling-system/
├── data/
│   ├── instances/           # Problem instances
│   └── results/            # Algorithm results
├── src/
│   ├── algorithms/         # Algorithm implementations
│   ├── models/            # Problem and solution models
│   ├── utils/             # Utility functions
│   └── experiments/       # Benchmark tools
├── database/
│   ├── database.py            # Database management
│   ├── database_adapter.py    # Database-algorithm bridge
│   ├── course_scheduling_gui.py # Main GUI application
│   ├── migrate_csv.py         # CSV migration tool
│   └── fix_imports.py       # FIxes imports
└── main.py               # Command-line interface
```

## Running the Project

### 1. GUI Application (Recommended)
Launch the graphical interface:
```bash
python course_scheduling_gui.py
```

Features:
- Create and manage problem instances
- Import data from CSV files
- Run optimization algorithms
- View results with web visualization
- Export results

### 2. Command-Line Interface
Run the main orchestrator:
```bash
python main.py                              # Interactive mode
python main.py --mode benchmark             # Run benchmarks
python main.py --mode single --algorithm greedy --instance small_01
```

### 3. Database Migration
Import existing CSV instances:
```bash
python migrate_csv.py --instance_path "data/instances/small/small_01"
python migrate_csv.py --migrate_all --instances_dir "data/instances"
```

## Usage Examples

### Quick Start
1. Run the GUI: `python course_scheduling_gui.py`
2. Create a new instance or import from CSV
3. Select an algorithm and configure parameters
4. Run the optimization
5. View results and visualizations


### Single Algorithm Run
```bash
python main.py --mode single --algorithm greedy --instance small_01 --verbose
```

## Algorithm Configuration
Each algorithm supports multiple variants and parameters:
- **Greedy**: Business optimal, balanced, fast fill variants
- **Genetic**: Standard, elite, adaptive variants
- **Simulated Annealing**: Classic, fast, slow variants
- **Tabu Search**: Classic, adaptive, fast variants
- **VNS**: Basic, skewed, general variants

## Data Format
The system supports:
- CSV files for courses, rooms, instructors, participants
- Compatibility matrices for instructor-course and room-course assignments
- Metadata files for instance information

## Results Analysis
- Automatic result saving in JSON format
- Performance metrics and statistical analysis
- Web-based schedule visualization
- CSV export for further analysis

## Common Issues and Solutions

1. **Import errors**:
   - Ensure all dependencies are installed
   - Check Python path configuration

2. **Database issues**:
   - Delete `course_scheduling.db` to reset database
   - Check file permissions

3. **GUI issues**:
   - Ensure tkinter is available (usually included with Python)
   - Try running from command line to see error messages

## Dependencies
See `requirements.txt` for full list of dependencies.

