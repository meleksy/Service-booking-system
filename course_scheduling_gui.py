#!/usr/bin/env python3
"""
Course Scheduling GUI - Complete Application
Part 1: Imports, Main Class and Widget Creation

Simple Tkinter-based GUI for the course scheduling optimization system.
Integrates with SQLite database and algorithm system.

Usage:
    python course_scheduling_gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
import threading
import queue
import logging
from pathlib import Path
from datetime import datetime

import tempfile
import webbrowser
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gui.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add database modules to path
sys.path.append(str(Path(__file__).parent))
from fix_imports import fix_algorithm_imports

fix_algorithm_imports()

from database import DatabaseManager
from database_adapter import DatabaseAdapter


class CourseSchedulingGUI:
    """Main GUI application for course scheduling system."""

    def __init__(self):
        """Initialize the GUI."""
        self.root = tk.Tk()
        self.root.title("Course Scheduling Optimization System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)

        # Initialize database
        self.db_path = "course_scheduling.db"
        self.db_manager = DatabaseManager(self.db_path)
        self.db_manager.initialize_database()
        self.adapter = DatabaseAdapter(self.db_manager)

        # Current instance
        self.current_instance_id = None
        self.current_instance_name = None

        # Selected items for editing
        self.selected_instructor_id = None
        self.selected_room_id = None
        self.selected_course_id = None

        # Threading support
        self.algorithm_thread = None
        self.progress_queue = queue.Queue()

        # ADD THIS LINE - Algorithm parameters storage
        self.algorithm_parameters = {}

        # Create GUI
        self.create_widgets()
        self.refresh_instance_list()

        # Start queue processor
        self.process_queue()

        logger.info("GUI initialized successfully")

    def create_widgets(self):
        """Create all GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Instance selection frame
        self.create_instance_frame(main_frame)

        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))

        # Create tabs
        self.create_data_tab()
        self.create_algorithm_tab()
        self.create_results_tab()

    def create_instance_frame(self, parent):
        """Create instance selection frame."""
        instance_frame = ttk.LabelFrame(parent, text="Instance Management", padding="10")
        instance_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # Instance selection
        ttk.Label(instance_frame, text="Current Instance:").grid(row=0, column=0, sticky=tk.W)

        self.instance_var = tk.StringVar()
        self.instance_combo = ttk.Combobox(instance_frame, textvariable=self.instance_var,
                                           state="readonly", width=30)
        self.instance_combo.grid(row=0, column=1, padx=(10, 0), sticky=(tk.W, tk.E))
        self.instance_combo.bind('<<ComboboxSelected>>', self.on_instance_selected)

        # Buttons
        ttk.Button(instance_frame, text="Refresh",
                   command=self.refresh_instance_list).grid(row=0, column=2, padx=(10, 0))
        ttk.Button(instance_frame, text="New Instance",
                   command=self.create_new_instance).grid(row=0, column=3, padx=(10, 0))
        ttk.Button(instance_frame, text="Import CSV",
                   command=self.import_csv_instance).grid(row=0, column=4, padx=(10, 0))

        # Configure grid weights
        instance_frame.columnconfigure(1, weight=1)

    def create_data_tab(self):
        """Create data management tab."""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="Data Management")

        # Create notebook for data types
        data_notebook = ttk.Notebook(data_frame)
        data_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Instructors tab
        self.create_instructors_tab(data_notebook)

        # Rooms tab
        self.create_rooms_tab(data_notebook)

        # Courses tab
        self.create_courses_tab(data_notebook)

        # Reservations tab
        self.create_reservations_tab(data_notebook)

    def create_instructors_tab(self, parent):
        """Create instructors management tab."""
        instructors_frame = ttk.Frame(parent)
        parent.add(instructors_frame, text="Instructors")

        # List frame
        list_frame = ttk.Frame(instructors_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5), pady=10)

        ttk.Label(list_frame, text="Instructors", font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W)

        # Instructors tree with scrollbar
        tree_frame = ttk.Frame(list_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.instructors_tree = ttk.Treeview(tree_frame, columns=('Name', 'Rate', 'Availability'), show='tree headings')
        self.instructors_tree.heading('#0', text='ID')
        self.instructors_tree.heading('Name', text='Name')
        self.instructors_tree.heading('Rate', text='Hourly Rate')
        self.instructors_tree.heading('Availability', text='Available Slots')

        self.instructors_tree.column('#0', width=50)
        self.instructors_tree.column('Name', width=150)
        self.instructors_tree.column('Rate', width=100)
        self.instructors_tree.column('Availability', width=200)

        # Scrollbar for instructors tree
        instructors_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.instructors_tree.yview)
        self.instructors_tree.configure(yscrollcommand=instructors_scrollbar.set)

        self.instructors_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        instructors_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.instructors_tree.bind('<Double-1>', self.edit_instructor)
        self.instructors_tree.bind('<ButtonRelease-1>', self.on_instructor_select)

        # Edit frame
        edit_frame = ttk.LabelFrame(instructors_frame, text="Edit Instructor", padding="10")
        edit_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 10), pady=10)

        # Instructor form
        ttk.Label(edit_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.instructor_name_var = tk.StringVar()
        ttk.Entry(edit_frame, textvariable=self.instructor_name_var, width=25).grid(row=0, column=1, pady=2)

        ttk.Label(edit_frame, text="Hourly Rate:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.instructor_rate_var = tk.StringVar()
        ttk.Entry(edit_frame, textvariable=self.instructor_rate_var, width=25).grid(row=1, column=1, pady=2)

        ttk.Label(edit_frame, text="Available Slots:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Label(edit_frame, text="(semicolon separated)").grid(row=3, column=0, columnspan=2, sticky=tk.W)
        self.instructor_slots_var = tk.StringVar()
        ttk.Entry(edit_frame, textvariable=self.instructor_slots_var, width=25).grid(row=4, column=1, pady=2)

        # Buttons
        button_frame = ttk.Frame(edit_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=15)

        ttk.Button(button_frame, text="Add New", command=self.add_instructor).pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Update", command=self.update_instructor).pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Delete", command=self.delete_instructor).pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Clear", command=self.clear_instructor_form).pack(side=tk.TOP, fill=tk.X, pady=2)

    def create_rooms_tab(self, parent):
        """Create rooms management tab."""
        rooms_frame = ttk.Frame(parent)
        parent.add(rooms_frame, text="Rooms")

        # List frame
        list_frame = ttk.Frame(rooms_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5), pady=10)

        ttk.Label(list_frame, text="Rooms", font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W)

        # Rooms tree with scrollbar
        tree_frame = ttk.Frame(list_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.rooms_tree = ttk.Treeview(tree_frame, columns=('Name', 'Capacity', 'Usage Cost', 'Idle Cost'),
                                       show='tree headings')
        self.rooms_tree.heading('#0', text='ID')
        self.rooms_tree.heading('Name', text='Name')
        self.rooms_tree.heading('Capacity', text='Capacity')
        self.rooms_tree.heading('Usage Cost', text='Usage Cost/h')
        self.rooms_tree.heading('Idle Cost', text='Idle Cost/h')

        self.rooms_tree.column('#0', width=50)
        self.rooms_tree.column('Name', width=150)
        self.rooms_tree.column('Capacity', width=80)
        self.rooms_tree.column('Usage Cost', width=100)
        self.rooms_tree.column('Idle Cost', width=100)

        # Scrollbar for rooms tree
        rooms_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.rooms_tree.yview)
        self.rooms_tree.configure(yscrollcommand=rooms_scrollbar.set)

        self.rooms_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        rooms_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.rooms_tree.bind('<Double-1>', self.edit_room)
        self.rooms_tree.bind('<ButtonRelease-1>', self.on_room_select)

        # Edit frame
        edit_frame = ttk.LabelFrame(rooms_frame, text="Edit Room", padding="10")
        edit_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 10), pady=10)

        # Room form
        ttk.Label(edit_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.room_name_var = tk.StringVar()
        ttk.Entry(edit_frame, textvariable=self.room_name_var, width=25).grid(row=0, column=1, pady=2)

        ttk.Label(edit_frame, text="Capacity:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.room_capacity_var = tk.StringVar()
        ttk.Entry(edit_frame, textvariable=self.room_capacity_var, width=25).grid(row=1, column=1, pady=2)

        ttk.Label(edit_frame, text="Usage Cost/h:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.room_usage_cost_var = tk.StringVar()
        ttk.Entry(edit_frame, textvariable=self.room_usage_cost_var, width=25).grid(row=2, column=1, pady=2)

        ttk.Label(edit_frame, text="Idle Cost/h:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.room_idle_cost_var = tk.StringVar()
        ttk.Entry(edit_frame, textvariable=self.room_idle_cost_var, width=25).grid(row=3, column=1, pady=2)

        # Buttons
        button_frame = ttk.Frame(edit_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=15)

        ttk.Button(button_frame, text="Add New", command=self.add_room).pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Update", command=self.update_room).pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Delete", command=self.delete_room).pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Clear", command=self.clear_room_form).pack(side=tk.TOP, fill=tk.X, pady=2)

    def create_courses_tab(self, parent):
        """Create courses management tab."""
        courses_frame = ttk.Frame(parent)
        parent.add(courses_frame, text="Courses")

        # List frame
        list_frame = ttk.Frame(courses_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5), pady=10)

        ttk.Label(list_frame, text="Courses", font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W)

        # Courses tree with scrollbar
        tree_frame = ttk.Frame(list_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.courses_tree = ttk.Treeview(tree_frame,
                                         columns=('Name', 'Duration', 'Price', 'Max Participants', 'Current'),
                                         show='tree headings')
        self.courses_tree.heading('#0', text='ID')
        self.courses_tree.heading('Name', text='Name')
        self.courses_tree.heading('Duration', text='Duration')
        self.courses_tree.heading('Price', text='Price/Person')
        self.courses_tree.heading('Max Participants', text='Max')
        self.courses_tree.heading('Current', text='Current')

        self.courses_tree.column('#0', width=50)
        self.courses_tree.column('Name', width=150)
        self.courses_tree.column('Duration', width=70)
        self.courses_tree.column('Price', width=80)
        self.courses_tree.column('Max Participants', width=60)
        self.courses_tree.column('Current', width=60)

        # Scrollbar for courses tree
        courses_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.courses_tree.yview)
        self.courses_tree.configure(yscrollcommand=courses_scrollbar.set)

        self.courses_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        courses_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.courses_tree.bind('<Double-1>', self.edit_course)
        self.courses_tree.bind('<ButtonRelease-1>', self.on_course_select)

        # Edit frame
        edit_frame = ttk.LabelFrame(courses_frame, text="Edit Course", padding="10")
        edit_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 10), pady=10)

        # Course form
        ttk.Label(edit_frame, text="Name:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.course_name_var = tk.StringVar()
        ttk.Entry(edit_frame, textvariable=self.course_name_var, width=25).grid(row=0, column=1, pady=2)

        ttk.Label(edit_frame, text="Duration:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.course_duration_var = tk.StringVar()
        ttk.Entry(edit_frame, textvariable=self.course_duration_var, width=25).grid(row=1, column=1, pady=2)

        ttk.Label(edit_frame, text="Price/Person:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.course_price_var = tk.StringVar()
        ttk.Entry(edit_frame, textvariable=self.course_price_var, width=25).grid(row=2, column=1, pady=2)

        ttk.Label(edit_frame, text="Max Participants:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.course_max_participants_var = tk.StringVar()
        ttk.Entry(edit_frame, textvariable=self.course_max_participants_var, width=25).grid(row=3, column=1, pady=2)

        ttk.Label(edit_frame, text="Current Participants:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.course_current_participants_var = tk.StringVar()
        ttk.Entry(edit_frame, textvariable=self.course_current_participants_var, width=25).grid(row=4, column=1, pady=2)

        # Buttons
        button_frame = ttk.Frame(edit_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=15)

        ttk.Button(button_frame, text="Add New", command=self.add_course).pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Update", command=self.update_course).pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Delete", command=self.delete_course).pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Clear", command=self.clear_course_form).pack(side=tk.TOP, fill=tk.X, pady=2)

    def create_algorithm_tab(self):
        """Create algorithm execution tab with dynamic variant selection."""
        algorithm_frame = ttk.Frame(self.notebook)
        self.notebook.add(algorithm_frame, text="Run Algorithms")

        # Algorithm selection frame
        selection_frame = ttk.LabelFrame(algorithm_frame, text="Algorithm Selection", padding="10")
        selection_frame.pack(fill=tk.X, padx=10, pady=10)

        # Algorithm selection
        ttk.Label(selection_frame, text="Algorithm:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.algorithm_var = tk.StringVar(value="GreedyAlgorithm")
        self.algorithm_combo = ttk.Combobox(selection_frame, textvariable=self.algorithm_var, values=[
            "GreedyAlgorithm", "GeneticAlgorithm", "SimulatedAnnealingAlgorithm",
            "TabuSearchAlgorithm", "VariableNeighborhoodSearch"
        ], state="readonly")
        self.algorithm_combo.grid(row=0, column=1, padx=(10, 0), sticky=(tk.W, tk.E))

        # Bind algorithm selection change to update variants
        self.algorithm_combo.bind('<<ComboboxSelected>>', self.on_algorithm_changed)

        # Variant selection
        ttk.Label(selection_frame, text="Variant:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.variant_var = tk.StringVar()
        self.variant_combo = ttk.Combobox(selection_frame, textvariable=self.variant_var, state="readonly")
        self.variant_combo.grid(row=1, column=1, padx=(10, 0), sticky=(tk.W, tk.E))

        # Parameters button
        self.params_button = ttk.Button(selection_frame, text="Configure Parameters",
                                        command=self.configure_algorithm_parameters, state="disabled")
        self.params_button.grid(row=2, column=1, padx=(10, 0), pady=5, sticky=tk.W)

        selection_frame.columnconfigure(1, weight=1)

        # Initialize variants for default algorithm
        self.on_algorithm_changed(None)

        # Run control frame
        run_frame = ttk.Frame(algorithm_frame)
        run_frame.pack(fill=tk.X, padx=10, pady=10)

        self.run_button = ttk.Button(run_frame, text="🚀 Run Algorithm", command=self.run_algorithm)
        self.run_button.pack(side=tk.LEFT)

        # Progress bar
        self.progress_bar = ttk.Progressbar(run_frame, mode='indeterminate')
        self.progress_bar.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)

        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(run_frame, textvariable=self.progress_var).pack(side=tk.RIGHT, padx=(10, 0))

        # Results preview
        preview_frame = ttk.LabelFrame(algorithm_frame, text="Last Result Preview", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Text with scrollbar
        text_frame = ttk.Frame(preview_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(text_frame, height=15, wrap=tk.WORD)
        result_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=result_scrollbar.set)

        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Store algorithm parameters
        self.algorithm_parameters = {}

    def get_algorithm_variants(self, algorithm_name):
        """Get available variants for specified algorithm."""
        try:
            if algorithm_name == "GreedyAlgorithm":
                # Try to import and get variants
                try:
                    # Fix import path
                    sys.path.append(str(Path(__file__).parent.parent / "src" / "algorithms"))
                    from greedy import create_greedy_variants
                    variants = create_greedy_variants()
                    return [v['name'] for v in variants]
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Could not import greedy variants: {e}")
                    # Fallback variants for Greedy
                    return ["Greedy_BusinessOptimal", "Greedy_Balanced", "Greedy_FastFill",
                            "Greedy_ProfitFirst", "Greedy_CapacityFirst"]

            elif algorithm_name == "GeneticAlgorithm":
                try:
                    sys.path.append(str(Path(__file__).parent.parent / "src" / "algorithms"))
                    from genetic import create_genetic_variants
                    variants = create_genetic_variants()
                    return [v['name'] for v in variants]
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Could not import genetic variants: {e}")
                    return ["GA_Standard", "GA_Elite", "GA_Adaptive", "GA_Aggressive", "GA_Conservative"]

            elif algorithm_name == "SimulatedAnnealingAlgorithm":
                try:
                    sys.path.append(str(Path(__file__).parent.parent / "src" / "algorithms"))
                    from simulated_annealing import create_sa_variants
                    variants = create_sa_variants()
                    return [v['name'] for v in variants]
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Could not import SA variants: {e}")
                    return ["SA_Classic", "SA_Fast", "SA_Slow", "SA_Adaptive", "SA_Geometric"]

            elif algorithm_name == "TabuSearchAlgorithm":
                try:
                    sys.path.append(str(Path(__file__).parent.parent / "src" / "algorithms"))
                    from tabu_search import create_tabu_variants
                    variants = create_tabu_variants()
                    return [v['name'] for v in variants]
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Could not import TS variants: {e}")
                    return ["TS_Classic", "TS_Adaptive", "TS_Fast", "TS_Elite", "TS_Aggressive"]

            elif algorithm_name == "VariableNeighborhoodSearch":
                try:
                    sys.path.append(str(Path(__file__).parent.parent / "src" / "algorithms"))
                    from vns import create_vns_variants
                    variants = create_vns_variants()
                    return [v['name'] for v in variants]
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Could not import VNS variants: {e}")
                    return ["VNS_Basic", "VNS_Skewed", "VNS_General", "VNS_Reduced", "VNS_Pipe"]

            else:
                return ["Default"]

        except Exception as e:
            logger.warning(f"Failed to get variants for {algorithm_name}: {e}")
            return ["Default"]

    def get_algorithm_variant_parameters(self, algorithm_name, variant_name):
        """Get parameters for specific algorithm variant."""
        try:
            if algorithm_name == "GreedyAlgorithm":
                try:
                    sys.path.append(str(Path(__file__).parent.parent / "src" / "algorithms"))
                    from greedy import create_greedy_variants
                    variants = create_greedy_variants()
                    variant = next((v for v in variants if v['name'] == variant_name), None)
                    if variant:
                        # Remove 'name' key to avoid conflicts
                        return {k: v for k, v in variant.items() if k != 'name'}
                    else:
                        return {}
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Could not import greedy parameters: {e}")
                    # Return default parameters based on variant name
                    if "BusinessOptimal" in variant_name:
                        return {
                            'course_selection_strategy': 'profit_density',
                            'room_selection_strategy': 'cheapest',
                            'instructor_selection_strategy': 'cheapest',
                            'participant_strategy': 'maximum'
                        }
                    elif "FastFill" in variant_name:
                        return {
                            'course_selection_strategy': 'first_available',
                            'room_selection_strategy': 'first_available',
                            'instructor_selection_strategy': 'first_available',
                            'participant_strategy': 'maximum'
                        }
                    else:
                        return {}

            elif algorithm_name == "TabuSearchAlgorithm":
                try:
                    sys.path.append(str(Path(__file__).parent.parent / "src" / "algorithms"))
                    from tabu_search import create_tabu_variants
                    variants = create_tabu_variants()
                    variant = next((v for v in variants if v['name'] == variant_name), None)
                    if variant:
                        # Remove 'name' key to avoid conflicts
                        return {k: v for k, v in variant.items() if k != 'name'}
                    else:
                        return {}
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Could not import TS parameters: {e}")
                    # Return default TS parameters
                    if "Classic" in variant_name:
                        return {
                            'tabu_tenure': 10,
                            'neighborhood_strategy': 'best_improvement',
                            'max_iterations': 1000
                        }
                    elif "Fast" in variant_name:
                        return {
                            'tabu_tenure': 7,
                            'neighborhood_strategy': 'first_improvement',
                            'max_iterations': 800
                        }
                    else:
                        return {}

            elif algorithm_name == "GeneticAlgorithm":
                try:
                    sys.path.append(str(Path(__file__).parent.parent / "src" / "algorithms"))
                    from genetic import create_genetic_variants
                    variants = create_genetic_variants()
                    variant = next((v for v in variants if v['name'] == variant_name), None)
                    if variant:
                        # Remove 'name' key to avoid conflicts
                        return {k: v for k, v in variant.items() if k != 'name'}
                    else:
                        return {}
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Could not import GA parameters: {e}")
                    # Default GA parameters based on variant
                    if "Standard" in variant_name:
                        return {
                            'population_size': 50,
                            'max_generations': 100,
                            'crossover_rate': 0.8,
                            'mutation_rate': 0.1
                        }
                    elif "Aggressive" in variant_name or "HighDiversity" in variant_name:
                        return {
                            'population_size': 100,
                            'max_generations': 200,
                            'crossover_rate': 0.9,
                            'mutation_rate': 0.2
                        }
                    else:
                        return {}

            elif algorithm_name == "SimulatedAnnealingAlgorithm":
                try:
                    sys.path.append(str(Path(__file__).parent.parent / "src" / "algorithms"))
                    from simulated_annealing import create_sa_variants
                    variants = create_sa_variants()
                    variant = next((v for v in variants if v['name'] == variant_name), None)
                    if variant:
                        # Remove 'name' key to avoid conflicts
                        return {k: v for k, v in variant.items() if k != 'name'}
                    else:
                        return {}
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Could not import SA parameters: {e}")
                    # Default SA parameters
                    if "Classic" in variant_name:
                        return {
                            'initial_temperature': 1000,
                            'final_temperature': 1,
                            'cooling_rate': 0.95,
                            'max_iterations': 1000
                        }
                    elif "Fast" in variant_name:
                        return {
                            'initial_temperature': 500,
                            'final_temperature': 1,
                            'cooling_rate': 0.90,
                            'max_iterations': 500
                        }
                    else:
                        return {}

            elif algorithm_name == "VariableNeighborhoodSearch":
                try:
                    sys.path.append(str(Path(__file__).parent.parent / "src" / "algorithms"))
                    from vns import create_vns_variants
                    variants = create_vns_variants()
                    variant = next((v for v in variants if v['name'] == variant_name), None)
                    if variant:
                        # Remove 'name' key to avoid conflicts
                        return {k: v for k, v in variant.items() if k != 'name'}
                    else:
                        return {}
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Could not import VNS parameters: {e}")
                    return {}

            else:
                return {}

        except Exception as e:
            logger.warning(f"Failed to get parameters for {algorithm_name} - {variant_name}: {e}")
            return {}

    def on_algorithm_changed(self, event):
        """Handle algorithm selection change."""
        algorithm_name = self.algorithm_var.get()

        # Get variants for selected algorithm
        variants = self.get_algorithm_variants(algorithm_name)

        # Update variant combobox
        self.variant_combo['values'] = variants

        # Select first variant by default
        if variants:
            self.variant_combo.current(0)
            variant_name = variants[0]

            # Load default parameters for this variant
            parameters = self.get_algorithm_variant_parameters(algorithm_name, variant_name)
            self.algorithm_parameters[algorithm_name] = {variant_name: parameters}

        # Enable parameters button
        self.params_button.config(state="normal")

    def configure_algorithm_parameters(self):
        """Open dialog to configure algorithm parameters."""
        algorithm_name = self.algorithm_var.get()
        variant_name = self.variant_var.get()

        if not algorithm_name or not variant_name:
            self.show_warning("Warning", "Please select algorithm and variant first")
            return

        # Get current parameters
        current_params = self.algorithm_parameters.get(algorithm_name, {}).get(variant_name, {})
        if not current_params:
            current_params = self.get_algorithm_variant_parameters(algorithm_name, variant_name)

        # Open parameters dialog
        dialog = AlgorithmParametersDialog(self.root, algorithm_name, variant_name, current_params)
        self.root.wait_window(dialog.dialog)

        if dialog.result:
            # Store updated parameters
            if algorithm_name not in self.algorithm_parameters:
                self.algorithm_parameters[algorithm_name] = {}
            self.algorithm_parameters[algorithm_name][variant_name] = dialog.result

            self.show_info("Parameters Updated", f"Parameters for {algorithm_name} - {variant_name} have been updated")

    def create_results_tab(self):
        """Create results viewing tab."""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="View Results")

        # Results list
        list_frame = ttk.LabelFrame(results_frame, text="Algorithm Runs", padding="10")
        list_frame.pack(fill=tk.X, padx=10, pady=10)

        # Results tree with scrollbar
        tree_frame = ttk.Frame(list_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.results_tree = ttk.Treeview(tree_frame, columns=('Algorithm', 'Variant', 'Objective', 'Feasible', 'Time'),
                                         show='tree headings', height=8)
        self.results_tree.heading('#0', text='Run ID')
        self.results_tree.heading('Algorithm', text='Algorithm')
        self.results_tree.heading('Variant', text='Variant')
        self.results_tree.heading('Objective', text='Objective')
        self.results_tree.heading('Feasible', text='Feasible')
        self.results_tree.heading('Time', text='Time (s)')

        self.results_tree.column('#0', width=80)
        self.results_tree.column('Algorithm', width=150)
        self.results_tree.column('Variant', width=120)
        self.results_tree.column('Objective', width=100)
        self.results_tree.column('Feasible', width=80)
        self.results_tree.column('Time', width=80)

        # Scrollbar for results tree
        results_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=results_scrollbar.set)

        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.results_tree.bind('<Double-1>', self.view_schedule)

        # Buttons for results
        results_buttons = ttk.Frame(list_frame)
        results_buttons.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(results_buttons, text="Refresh", command=self.refresh_results).pack(side=tk.LEFT)
        ttk.Button(results_buttons, text="Delete Selected", command=self.delete_selected_result).pack(side=tk.LEFT,
                                                                                                      padx=(10, 0))
        # Dodaj ten kod po linii z przyciskiem "Delete Selected"
        ttk.Button(results_buttons, text="🌐 Web Visualization",
                   command=self.show_web_visualization).pack(side=tk.LEFT, padx=(10, 0))

        # Schedule grid
        grid_frame = ttk.LabelFrame(results_frame, text="Schedule Grid", padding="10")
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Grid text with scrollbars
        grid_text_frame = ttk.Frame(grid_frame)
        grid_text_frame.pack(fill=tk.BOTH, expand=True)

        self.grid_text = tk.Text(grid_text_frame, height=15, wrap=tk.NONE, font=('Courier', 9))
        grid_scrollbar_y = ttk.Scrollbar(grid_text_frame, orient=tk.VERTICAL, command=self.grid_text.yview)
        grid_scrollbar_x = ttk.Scrollbar(grid_text_frame, orient=tk.HORIZONTAL, command=self.grid_text.xview)
        self.grid_text.configure(yscrollcommand=grid_scrollbar_y.set, xscrollcommand=grid_scrollbar_x.set)

        self.grid_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        grid_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        grid_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

    def create_reservations_tab(self, parent):
        """Create reservations management tab."""
        reservations_frame = ttk.Frame(parent)
        parent.add(reservations_frame, text="Reservations")

        # Clients section
        clients_frame = ttk.LabelFrame(reservations_frame, text="Clients", padding="10")
        clients_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        # Client management controls
        clients_controls = ttk.Frame(clients_frame)
        clients_controls.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(clients_controls, text="Add Client", command=self.add_client_dialog).pack(side=tk.LEFT)
        ttk.Button(clients_controls, text="Refresh Clients", command=self.refresh_clients).pack(side=tk.LEFT,
                                                                                                padx=(10, 0))

        # Clients list
        clients_tree_frame = ttk.Frame(clients_frame)
        clients_tree_frame.pack(fill=tk.X)

        self.clients_tree = ttk.Treeview(clients_tree_frame, columns=('Name', 'Email', 'Phone'),
                                         show='tree headings', height=4)
        self.clients_tree.heading('#0', text='ID')
        self.clients_tree.heading('Name', text='Name')
        self.clients_tree.heading('Email', text='Email')
        self.clients_tree.heading('Phone', text='Phone')

        self.clients_tree.column('#0', width=50)
        self.clients_tree.column('Name', width=200)
        self.clients_tree.column('Email', width=200)
        self.clients_tree.column('Phone', width=120)

        clients_scrollbar = ttk.Scrollbar(clients_tree_frame, orient=tk.VERTICAL, command=self.clients_tree.yview)
        self.clients_tree.configure(yscrollcommand=clients_scrollbar.set)

        self.clients_tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        clients_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Reservations section
        reservations_main_frame = ttk.LabelFrame(reservations_frame, text="Reservations", padding="10")
        reservations_main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Reservation controls
        res_controls = ttk.Frame(reservations_main_frame)
        res_controls.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(res_controls, text="New Reservation", command=self.add_reservation_dialog).pack(side=tk.LEFT)
        ttk.Button(res_controls, text="Refresh", command=self.refresh_reservations).pack(side=tk.LEFT, padx=(10, 0))

        # Status filter
        ttk.Label(res_controls, text="Status:").pack(side=tk.LEFT, padx=(20, 5))
        self.reservation_status_filter = tk.StringVar(value="all")
        status_combo = ttk.Combobox(res_controls, textvariable=self.reservation_status_filter,
                                    values=["all", "pending", "confirmed", "cancelled"],
                                    state="readonly", width=12)
        status_combo.pack(side=tk.LEFT)
        status_combo.bind('<<ComboboxSelected>>', lambda e: self.refresh_reservations())

        # Reservations list
        res_tree_frame = ttk.Frame(reservations_main_frame)
        res_tree_frame.pack(fill=tk.BOTH, expand=True)

        self.reservations_tree = ttk.Treeview(res_tree_frame,
                                              columns=('Client', 'Course', 'Status', 'Date', 'Notes'),
                                              show='tree headings')
        self.reservations_tree.heading('#0', text='ID')
        self.reservations_tree.heading('Client', text='Client')
        self.reservations_tree.heading('Course', text='Course')
        self.reservations_tree.heading('Status', text='Status')
        self.reservations_tree.heading('Date', text='Date')
        self.reservations_tree.heading('Notes', text='Notes')

        self.reservations_tree.column('#0', width=50)
        self.reservations_tree.column('Client', width=150)
        self.reservations_tree.column('Course', width=150)
        self.reservations_tree.column('Status', width=100)
        self.reservations_tree.column('Date', width=120)
        self.reservations_tree.column('Notes', width=200)

        res_scrollbar = ttk.Scrollbar(res_tree_frame, orient=tk.VERTICAL, command=self.reservations_tree.yview)
        self.reservations_tree.configure(yscrollcommand=res_scrollbar.set)

        self.reservations_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        res_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Reservation actions
        res_actions = ttk.Frame(reservations_main_frame)
        res_actions.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(res_actions, text="✓ Confirm", command=self.confirm_reservation).pack(side=tk.LEFT)
        ttk.Button(res_actions, text="✗ Cancel", command=self.cancel_reservation).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(res_actions, text="📝 Edit", command=self.edit_reservation_dialog).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(res_actions, text="🗑 Delete", command=self.delete_reservation).pack(side=tk.LEFT, padx=(10, 0))

        # Bind double-click to edit
        self.reservations_tree.bind('<Double-1>', self.edit_reservation_dialog)

    # Clients management methods
    def refresh_clients(self):
        """Refresh clients list."""
        try:
            # Clear tree
            for item in self.clients_tree.get_children():
                self.clients_tree.delete(item)

            # Load clients
            clients = self.db_manager.get_clients()

            for client in clients:
                self.clients_tree.insert('', tk.END,
                                         text=str(client['id']),
                                         values=(
                                             f"{client['first_name']} {client['last_name']}",
                                             client['email'] or '',
                                             client['phone'] or ''
                                         ))

            logger.info(f"Refreshed {len(clients)} clients")

        except Exception as e:
            self.show_error("Error", f"Failed to load clients: {e}")
            logger.error(f"Failed to refresh clients: {e}")

    def add_client_dialog(self):
        """Show dialog to add new client."""
        dialog = ClientDialog(self.root, title="Add New Client")
        self.root.wait_window(dialog.dialog)

        if dialog.result:
            try:
                client_id = self.db_manager.add_client(
                    first_name=dialog.result['first_name'],
                    last_name=dialog.result['last_name'],
                    email=dialog.result['email'],
                    phone=dialog.result['phone']
                )

                self.show_info("Success", f"Added client '{dialog.result['first_name']} {dialog.result['last_name']}'")
                self.refresh_clients()

                logger.info(f"Added client with ID {client_id}")

            except Exception as e:
                self.show_error("Error", f"Failed to add client: {e}")
                logger.error(f"Failed to add client: {e}")

    # Reservations management methods
    def refresh_reservations(self):
        """Refresh reservations list."""
        if not self.current_instance_id:
            return

        try:
            # Clear tree
            for item in self.reservations_tree.get_children():
                self.reservations_tree.delete(item)

            # Get filter status
            status_filter = self.reservation_status_filter.get()
            status = None if status_filter == "all" else status_filter

            # Load reservations
            reservations = self.db_manager.get_reservations(self.current_instance_id, status)

            for reservation in reservations:
                # Format date
                date_str = reservation['reservation_date'][:10] if reservation['reservation_date'] else ''

                self.reservations_tree.insert('', tk.END,
                                              text=str(reservation['reservation_id']),
                                              values=(
                                                  reservation['client_name'] or 'Unknown',
                                                  reservation['course_name'] or f"Course {reservation['course_id']}",
                                                  reservation['status'].title(),
                                                  date_str,
                                                  reservation['notes'] or ''
                                              ))

            logger.info(f"Refreshed {len(reservations)} reservations")

        except Exception as e:
            self.show_error("Error", f"Failed to load reservations: {e}")
            logger.error(f"Failed to refresh reservations: {e}")

    def add_reservation_dialog(self):
        """Show dialog to add new reservation."""
        if not self.check_instance_selected():
            return

        # Get clients and courses for the dialog
        try:
            clients = self.db_manager.get_clients()
            courses = self.db_manager.get_courses(self.current_instance_id)

            if not clients:
                self.show_warning("Warning", "Please add clients first")
                return

            if not courses:
                self.show_warning("Warning", "No courses available in this instance")
                return

            dialog = ReservationDialog(self.root, clients, courses, title="New Reservation")
            self.root.wait_window(dialog.dialog)

            if dialog.result:
                reservation_id = self.db_manager.add_reservation(
                    instance_id=self.current_instance_id,
                    client_id=dialog.result['client_id'],
                    course_id=dialog.result['course_id'],
                    status=dialog.result['status'],
                    notes=dialog.result['notes']
                )

                self.show_info("Success", f"Created reservation #{reservation_id}")
                self.refresh_reservations()

                logger.info(f"Added reservation {reservation_id}")

        except Exception as e:
            self.show_error("Error", f"Failed to create reservation: {e}")
            logger.error(f"Failed to add reservation: {e}")

    def confirm_reservation(self):
        """Confirm selected reservation."""
        self._update_reservation_status('confirmed')

    def cancel_reservation(self):
        """Cancel selected reservation."""
        self._update_reservation_status('cancelled')

    def _update_reservation_status(self, new_status):
        """Update status of selected reservation."""
        selection = self.reservations_tree.selection()
        if not selection:
            self.show_warning("Warning", "Please select a reservation")
            return

        item = self.reservations_tree.item(selection[0])
        reservation_id = int(item['text'])

        try:
            success = self.db_manager.update_reservation(
                reservation_id=reservation_id,
                status=new_status
            )

            if success:
                self.show_info("Success", f"Reservation status updated to {new_status}")
                self.refresh_reservations()
                logger.info(f"Updated reservation {reservation_id} status to {new_status}")
            else:
                self.show_error("Error", "Failed to update reservation")

        except Exception as e:
            self.show_error("Error", f"Failed to update reservation: {e}")
            logger.error(f"Failed to update reservation {reservation_id}: {e}")

    def edit_reservation_dialog(self, event=None):
        """Edit selected reservation."""
        selection = self.reservations_tree.selection()
        if not selection:
            self.show_warning("Warning", "Please select a reservation to edit")
            return

        # Implementation for edit dialog would go here
        self.show_info("Info", "Edit reservation functionality - to be implemented")

    def delete_reservation(self):
        """Delete selected reservation."""
        selection = self.reservations_tree.selection()
        if not selection:
            self.show_warning("Warning", "Please select a reservation to delete")
            return

        item = self.reservations_tree.item(selection[0])
        reservation_id = int(item['text'])
        client_name = item['values'][0]

        if self.show_confirmation("Confirm Delete",
                                  f"Are you sure you want to delete reservation for '{client_name}'?"):
            try:
                success = self.db_manager.delete_reservation(reservation_id)

                if success:
                    self.show_info("Success", "Reservation deleted")
                    self.refresh_reservations()
                    logger.info(f"Deleted reservation {reservation_id}")
                else:
                    self.show_error("Error", "Failed to delete reservation")

            except Exception as e:
                self.show_error("Error", f"Failed to delete reservation: {e}")
                logger.error(f"Failed to delete reservation {reservation_id}: {e}")

    def run(self):
        """Start the GUI."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        """Handle application closing."""
        # Stop any running algorithm
        if self.algorithm_thread and self.algorithm_thread.is_alive():
            logger.info("Waiting for algorithm to finish...")
            # In a real implementation, you might want to add a flag to stop the algorithm

        logger.info("Application closing")
        self.root.destroy()

    def process_queue(self):
        """Process messages from background threads."""
        try:
            while True:
                message = self.progress_queue.get_nowait()
                self.progress_var.set(message)
                self.root.update_idletasks()
        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self.process_queue)

        # ============================================================================
        # INSTANCE MANAGEMENT METHODS
        # ============================================================================

    def refresh_instance_list(self):
        """Refresh the list of instances."""
        try:
            instances = self.db_manager.get_instances()
            instance_names = [f"{inst['name']} (ID: {inst['id']})" for inst in instances]

            self.instance_combo['values'] = instance_names

            if instances and not self.current_instance_id:
                # Select first instance by default
                self.instance_combo.current(0)
                self.on_instance_selected(None)

            logger.info(f"Refreshed instance list: {len(instances)} instances")

        except Exception as e:
            self.show_error("Error", f"Failed to load instances: {e}")
            logger.error(f"Failed to refresh instance list: {e}")

    def on_instance_selected(self, event):
        """Handle instance selection."""
        selection = self.instance_var.get()
        if not selection:
            return

        try:
            # Extract instance ID from selection
            instance_id = int(selection.split("ID: ")[1].split(")")[0])

            # Get instance details
            instances = self.db_manager.get_instances()
            instance = next((inst for inst in instances if inst['id'] == instance_id), None)

            if instance:
                self.current_instance_id = instance_id
                self.current_instance_name = instance['name']

                # Clear selected items
                self.clear_all_selections()

                # Refresh all data
                self.refresh_all_data()

                # Update status
                self.progress_var.set(f"Instance: {instance['name']}")

                logger.info(f"Selected instance: {instance['name']} (ID: {instance_id})")

        except Exception as e:
            self.show_error("Error", f"Failed to select instance: {e}")
            logger.error(f"Failed to select instance: {e}")

    def create_new_instance(self):
        """Create a new instance."""
        dialog = NewInstanceDialog(self.root, title="Create New Instance")
        self.root.wait_window(dialog.dialog)

        if dialog.result:
            try:
                instance_id = self.db_manager.create_instance(
                    name=dialog.result['name'],
                    description=dialog.result['description'],
                    time_horizon=dialog.result['time_horizon'],
                    difficulty=dialog.result['difficulty']
                )

                self.show_info("Success", f"Created instance '{dialog.result['name']}' with ID {instance_id}")
                self.refresh_instance_list()

                # Select the new instance
                for i, value in enumerate(self.instance_combo['values']):
                    if f"ID: {instance_id}" in value:
                        self.instance_combo.current(i)
                        self.on_instance_selected(None)
                        break

                logger.info(f"Created new instance: {dialog.result['name']} (ID: {instance_id})")

            except Exception as e:
                self.show_error("Error", f"Failed to create instance: {e}")
                logger.error(f"Failed to create instance: {e}")

    def import_csv_instance(self):
        """Import instance from CSV directory."""
        directory = filedialog.askdirectory(title="Select Instance Directory")
        if not directory:
            return

        try:
            from migrate_csv import CSVMigrator
            migrator = CSVMigrator(self.db_path)

            self.progress_var.set("Importing CSV...")
            self.progress_bar.start()
            self.root.update()

            instance_id = migrator.migrate_instance(directory, force=True)

            if instance_id:
                self.show_info("Success", f"Imported instance with ID {instance_id}")
                self.refresh_instance_list()

                # Select the imported instance
                for i, value in enumerate(self.instance_combo['values']):
                    if f"ID: {instance_id}" in value:
                        self.instance_combo.current(i)
                        self.on_instance_selected(None)
                        break

                logger.info(f"Imported CSV instance from {directory} (ID: {instance_id})")
            else:
                self.show_error("Error", "Failed to import CSV instance")

        except Exception as e:
            self.show_error("Error", f"Failed to import CSV: {e}")
            logger.error(f"Failed to import CSV from {directory}: {e}")
        finally:
            self.progress_bar.stop()
            self.progress_var.set("Ready")

    def refresh_all_data(self):
        """Refresh all data in the current instance."""
        if not self.current_instance_id:
            return

        try:
            self.refresh_instructors()
            self.refresh_rooms()
            self.refresh_courses()
            self.refresh_results()
            self.refresh_clients()
            self.refresh_reservations()
            logger.info(f"Refreshed all data for instance {self.current_instance_id}")
        except Exception as e:
            self.show_error("Error", f"Failed to refresh data: {e}")
            logger.error(f"Failed to refresh all data: {e}")

    def clear_all_selections(self):
        """Clear all selected items."""
        self.selected_instructor_id = None
        self.selected_room_id = None
        self.selected_course_id = None
        self.clear_instructor_form()
        self.clear_room_form()
        self.clear_course_form()

        # Clear reservation selections if trees exist
        if hasattr(self, 'clients_tree'):
            for item in self.clients_tree.selection():
                self.clients_tree.selection_remove(item)
        if hasattr(self, 'reservations_tree'):
            for item in self.reservations_tree.selection():
                self.reservations_tree.selection_remove(item)

    # ============================================================================
    # INSTRUCTOR MANAGEMENT METHODS
    # ============================================================================

    def refresh_instructors(self):
        """Refresh instructors list."""
        if not self.current_instance_id:
            return

        try:
            # Clear tree
            for item in self.instructors_tree.get_children():
                self.instructors_tree.delete(item)

            # Load instructors
            instructors = self.db_manager.get_instructors(self.current_instance_id)

            for instructor in instructors:
                self.instructors_tree.insert('', tk.END,
                                             text=str(instructor['instructor_id']),
                                             values=(
                                                 instructor['name'],
                                                 f"${instructor['hourly_rate']:.2f}",
                                                 instructor['available_slots'] or "All slots"
                                             ))

            logger.info(f"Refreshed {len(instructors)} instructors")

        except Exception as e:
            self.show_error("Error", f"Failed to load instructors: {e}")
            logger.error(f"Failed to refresh instructors: {e}")

    def on_instructor_select(self, event):
        """Handle instructor selection (single click)."""
        selection = self.instructors_tree.selection()
        if selection:
            item = self.instructors_tree.item(selection[0])
            self.selected_instructor_id = int(item['text'])

    def edit_instructor(self, event):
        """Edit selected instructor (double click)."""
        selection = self.instructors_tree.selection()
        if not selection:
            return

        item = self.instructors_tree.item(selection[0])
        instructor_id = int(item['text'])

        try:
            instructors = self.db_manager.get_instructors(self.current_instance_id)
            instructor = next((inst for inst in instructors if inst['instructor_id'] == instructor_id), None)

            if instructor:
                self.selected_instructor_id = instructor_id
                self.instructor_name_var.set(instructor['name'])
                self.instructor_rate_var.set(str(instructor['hourly_rate']))
                self.instructor_slots_var.set(instructor['available_slots'] or '')

                logger.info(f"Editing instructor {instructor_id}: {instructor['name']}")

        except Exception as e:
            self.show_error("Error", f"Failed to load instructor: {e}")
            logger.error(f"Failed to edit instructor {instructor_id}: {e}")

    def add_instructor(self):
        """Add new instructor."""
        if not self.check_instance_selected():
            return

        try:
            # Validate data
            name = self.instructor_name_var.get().strip()
            rate_str = self.instructor_rate_var.get().strip()
            slots = self.instructor_slots_var.get().strip()

            if not name or not rate_str:
                self.show_warning("Warning", "Name and hourly rate are required")
                return

            rate = float(rate_str)
            if rate <= 0:
                self.show_warning("Warning", "Hourly rate must be positive")
                return

            # Validate slots if provided
            if slots and not self.validate_time_slots(slots):
                self.show_warning("Warning",
                                  "Invalid time slots format. Use semicolon-separated numbers (e.g., 1;2;3)")
                return

            # Get next instructor ID
            instructors = self.db_manager.get_instructors(self.current_instance_id)
            next_id = max([inst['instructor_id'] for inst in instructors], default=0) + 1

            self.db_manager.add_instructor(
                instance_id=self.current_instance_id,
                instructor_id=next_id,
                name=name,
                hourly_rate=rate,
                available_slots=slots
            )

            self.show_info("Success", f"Added instructor '{name}'")
            self.clear_instructor_form()
            self.refresh_instructors()

            logger.info(f"Added instructor {next_id}: {name}")

        except ValueError:
            self.show_error("Error", "Invalid hourly rate value")
        except Exception as e:
            self.show_error("Error", f"Failed to add instructor: {e}")
            logger.error(f"Failed to add instructor: {e}")

    def update_instructor(self):
        """Update selected instructor."""
        if not self.selected_instructor_id:
            self.show_warning("Warning", "Please select an instructor to update")
            return

        try:
            # Validate data
            name = self.instructor_name_var.get().strip()
            rate_str = self.instructor_rate_var.get().strip()
            slots = self.instructor_slots_var.get().strip()

            if not name or not rate_str:
                self.show_warning("Warning", "Name and hourly rate are required")
                return

            rate = float(rate_str)
            if rate <= 0:
                self.show_warning("Warning", "Hourly rate must be positive")
                return

            # Validate slots if provided
            if slots and not self.validate_time_slots(slots):
                self.show_warning("Warning",
                                  "Invalid time slots format. Use semicolon-separated numbers (e.g., 1;2;3)")
                return

            success = self.db_manager.update_instructor(
                instance_id=self.current_instance_id,
                instructor_id=self.selected_instructor_id,
                name=name,
                hourly_rate=rate,
                available_slots=slots
            )

            if success:
                self.show_info("Success", f"Updated instructor '{name}'")
                self.clear_instructor_form()
                self.refresh_instructors()
                logger.info(f"Updated instructor {self.selected_instructor_id}: {name}")
            else:
                self.show_error("Error", "Failed to update instructor")

        except ValueError:
            self.show_error("Error", "Invalid hourly rate value")
        except Exception as e:
            self.show_error("Error", f"Failed to update instructor: {e}")
            logger.error(f"Failed to update instructor {self.selected_instructor_id}: {e}")

    def delete_instructor(self):
        """Delete selected instructor."""
        if not self.selected_instructor_id:
            self.show_warning("Warning", "Please select an instructor to delete")
            return

        # Get instructor name for confirmation
        try:
            instructors = self.db_manager.get_instructors(self.current_instance_id)
            instructor = next(
                (inst for inst in instructors if inst['instructor_id'] == self.selected_instructor_id), None)
            instructor_name = instructor['name'] if instructor else f"ID {self.selected_instructor_id}"
        except:
            instructor_name = f"ID {self.selected_instructor_id}"

        if self.show_confirmation("Confirm Delete",
                                  f"Are you sure you want to delete instructor '{instructor_name}'?"):
            try:
                success = self.db_manager.delete_instructor(
                    instance_id=self.current_instance_id,
                    instructor_id=self.selected_instructor_id
                )

                if success:
                    self.show_info("Success", f"Deleted instructor '{instructor_name}'")
                    self.clear_instructor_form()
                    self.refresh_instructors()
                    logger.info(f"Deleted instructor {self.selected_instructor_id}: {instructor_name}")
                else:
                    self.show_error("Error", "Failed to delete instructor")

            except Exception as e:
                self.show_error("Error", f"Failed to delete instructor: {e}")
                logger.error(f"Failed to delete instructor {self.selected_instructor_id}: {e}")

    def clear_instructor_form(self):
        """Clear instructor form."""
        self.selected_instructor_id = None
        self.instructor_name_var.set('')
        self.instructor_rate_var.set('')
        self.instructor_slots_var.set('')

    # ============================================================================
    # ROOM MANAGEMENT METHODS
    # ============================================================================

    def refresh_rooms(self):
        """Refresh rooms list."""
        if not self.current_instance_id:
            return

        try:
            # Clear tree
            for item in self.rooms_tree.get_children():
                self.rooms_tree.delete(item)

            # Load rooms
            rooms = self.db_manager.get_rooms(self.current_instance_id)

            for room in rooms:
                self.rooms_tree.insert('', tk.END,
                                       text=str(room['room_id']),
                                       values=(
                                           room['name'],
                                           str(room['capacity']),
                                           f"${room['usage_cost_per_hour']:.2f}",
                                           f"${room['idle_cost_per_hour']:.2f}"
                                       ))

            logger.info(f"Refreshed {len(rooms)} rooms")

        except Exception as e:
            self.show_error("Error", f"Failed to load rooms: {e}")
            logger.error(f"Failed to refresh rooms: {e}")

    def on_room_select(self, event):
        """Handle room selection (single click)."""
        selection = self.rooms_tree.selection()
        if selection:
            item = self.rooms_tree.item(selection[0])
            self.selected_room_id = int(item['text'])

    def edit_room(self, event):
        """Edit selected room (double click)."""
        selection = self.rooms_tree.selection()
        if not selection:
            return

        item = self.rooms_tree.item(selection[0])
        room_id = int(item['text'])

        try:
            rooms = self.db_manager.get_rooms(self.current_instance_id)
            room = next((r for r in rooms if r['room_id'] == room_id), None)

            if room:
                self.selected_room_id = room_id
                self.room_name_var.set(room['name'])
                self.room_capacity_var.set(str(room['capacity']))
                self.room_usage_cost_var.set(str(room['usage_cost_per_hour']))
                self.room_idle_cost_var.set(str(room['idle_cost_per_hour']))

                logger.info(f"Editing room {room_id}: {room['name']}")

        except Exception as e:
            self.show_error("Error", f"Failed to load room: {e}")
            logger.error(f"Failed to edit room {room_id}: {e}")

    def add_room(self):
        """Add new room."""
        if not self.check_instance_selected():
            return

        try:
            # Validate data
            name = self.room_name_var.get().strip()
            capacity_str = self.room_capacity_var.get().strip()
            usage_cost_str = self.room_usage_cost_var.get().strip()
            idle_cost_str = self.room_idle_cost_var.get().strip()

            if not name or not capacity_str:
                self.show_warning("Warning", "Name and capacity are required")
                return

            capacity = int(capacity_str)
            if capacity <= 0:
                self.show_warning("Warning", "Capacity must be positive")
                return

            usage_cost = float(usage_cost_str) if usage_cost_str else 0.0
            idle_cost = float(idle_cost_str) if idle_cost_str else 0.0

            if usage_cost < 0 or idle_cost < 0:
                self.show_warning("Warning", "Costs cannot be negative")
                return

            # Get next room ID
            rooms = self.db_manager.get_rooms(self.current_instance_id)
            next_id = max([room['room_id'] for room in rooms], default=0) + 1

            self.db_manager.add_room(
                instance_id=self.current_instance_id,
                room_id=next_id,
                name=name,
                capacity=capacity,
                usage_cost_per_hour=usage_cost,
                idle_cost_per_hour=idle_cost
            )

            self.show_info("Success", f"Added room '{name}'")
            self.clear_room_form()
            self.refresh_rooms()

            logger.info(f"Added room {next_id}: {name}")

        except ValueError as e:
            self.show_error("Error", "Invalid numeric values")
        except Exception as e:
            self.show_error("Error", f"Failed to add room: {e}")
            logger.error(f"Failed to add room: {e}")

    def update_room(self):
        """Update selected room."""
        if not self.selected_room_id:
            self.show_warning("Warning", "Please select a room to update")
            return

        try:
            # Validate data
            name = self.room_name_var.get().strip()
            capacity_str = self.room_capacity_var.get().strip()
            usage_cost_str = self.room_usage_cost_var.get().strip()
            idle_cost_str = self.room_idle_cost_var.get().strip()

            if not name or not capacity_str:
                self.show_warning("Warning", "Name and capacity are required")
                return

            capacity = int(capacity_str)
            if capacity <= 0:
                self.show_warning("Warning", "Capacity must be positive")
                return

            usage_cost = float(usage_cost_str) if usage_cost_str else 0.0
            idle_cost = float(idle_cost_str) if idle_cost_str else 0.0

            if usage_cost < 0 or idle_cost < 0:
                self.show_warning("Warning", "Costs cannot be negative")
                return

            success = self.db_manager.update_room(
                instance_id=self.current_instance_id,
                room_id=self.selected_room_id,
                name=name,
                capacity=capacity,
                usage_cost_per_hour=usage_cost,
                idle_cost_per_hour=idle_cost
            )

            if success:
                self.show_info("Success", f"Updated room '{name}'")
                self.clear_room_form()
                self.refresh_rooms()
                logger.info(f"Updated room {self.selected_room_id}: {name}")
            else:
                self.show_error("Error", "Failed to update room")

        except ValueError:
            self.show_error("Error", "Invalid numeric values")
        except Exception as e:
            self.show_error("Error", f"Failed to update room: {e}")
            logger.error(f"Failed to update room {self.selected_room_id}: {e}")

    def delete_room(self):
        """Delete selected room."""
        if not self.selected_room_id:
            self.show_warning("Warning", "Please select a room to delete")
            return

        # Get room name for confirmation
        try:
            rooms = self.db_manager.get_rooms(self.current_instance_id)
            room = next((r for r in rooms if r['room_id'] == self.selected_room_id), None)
            room_name = room['name'] if room else f"ID {self.selected_room_id}"
        except:
            room_name = f"ID {self.selected_room_id}"

        if self.show_confirmation("Confirm Delete", f"Are you sure you want to delete room '{room_name}'?"):
            try:
                success = self.db_manager.delete_room(
                    instance_id=self.current_instance_id,
                    room_id=self.selected_room_id
                )

                if success:
                    self.show_info("Success", f"Deleted room '{room_name}'")
                    self.clear_room_form()
                    self.refresh_rooms()
                    logger.info(f"Deleted room {self.selected_room_id}: {room_name}")
                else:
                    self.show_error("Error", "Failed to delete room")

            except Exception as e:
                self.show_error("Error", f"Failed to delete room: {e}")
                logger.error(f"Failed to delete room {self.selected_room_id}: {e}")

    def clear_room_form(self):
        """Clear room form."""
        self.selected_room_id = None
        self.room_name_var.set('')
        self.room_capacity_var.set('')
        self.room_usage_cost_var.set('')
        self.room_idle_cost_var.set('')

    # ============================================================================
    # COURSE MANAGEMENT METHODS
    # ============================================================================

    def refresh_courses(self):
        """Refresh courses list."""
        if not self.current_instance_id:
            return

        try:
            # Clear tree
            for item in self.courses_tree.get_children():
                self.courses_tree.delete(item)

            # Load courses
            courses = self.db_manager.get_courses(self.current_instance_id)

            for course in courses:
                self.courses_tree.insert('', tk.END,
                                         text=str(course['course_id']),
                                         values=(
                                             course['name'],
                                             str(course['duration']),
                                             f"${course['price_per_participant']:.2f}",
                                             str(course['max_participants']),
                                             str(course['current_participants'])
                                         ))

            logger.info(f"Refreshed {len(courses)} courses")

        except Exception as e:
            self.show_error("Error", f"Failed to load courses: {e}")
            logger.error(f"Failed to refresh courses: {e}")

    def on_course_select(self, event):
        """Handle course selection (single click)."""
        selection = self.courses_tree.selection()
        if selection:
            item = self.courses_tree.item(selection[0])
            self.selected_course_id = int(item['text'])

    def edit_course(self, event):
        """Edit selected course (double click)."""
        selection = self.courses_tree.selection()
        if not selection:
            return

        item = self.courses_tree.item(selection[0])
        course_id = int(item['text'])

        try:
            courses = self.db_manager.get_courses(self.current_instance_id)
            course = next((c for c in courses if c['course_id'] == course_id), None)

            if course:
                self.selected_course_id = course_id
                self.course_name_var.set(course['name'])
                self.course_duration_var.set(str(course['duration']))
                self.course_price_var.set(str(course['price_per_participant']))
                self.course_max_participants_var.set(str(course['max_participants']))
                self.course_current_participants_var.set(str(course['current_participants']))

                logger.info(f"Editing course {course_id}: {course['name']}")

        except Exception as e:
            self.show_error("Error", f"Failed to load course: {e}")
            logger.error(f"Failed to edit course {course_id}: {e}")

    def add_course(self):
        """Add new course."""
        if not self.check_instance_selected():
            return

        try:
            # Validate data
            name = self.course_name_var.get().strip()
            duration_str = self.course_duration_var.get().strip()
            price_str = self.course_price_var.get().strip()
            max_participants_str = self.course_max_participants_var.get().strip()
            current_participants_str = self.course_current_participants_var.get().strip()

            if not all([name, duration_str, price_str, max_participants_str]):
                self.show_warning("Warning", "Name, duration, price, and max participants are required")
                return

            duration = int(duration_str)
            if duration <= 0:
                self.show_warning("Warning", "Duration must be positive")
                return

            price = float(price_str)
            if price <= 0:
                self.show_warning("Warning", "Price must be positive")
                return

            max_participants = int(max_participants_str)
            if max_participants <= 0:
                self.show_warning("Warning", "Max participants must be positive")
                return

            current_participants = int(current_participants_str) if current_participants_str else max_participants
            if current_participants < 0 or current_participants > max_participants:
                self.show_warning("Warning", "Current participants must be between 0 and max participants")
                return

            # Get next course ID
            courses = self.db_manager.get_courses(self.current_instance_id)
            next_id = max([course['course_id'] for course in courses], default=0) + 1

            # Add course
            self.db_manager.add_course(
                instance_id=self.current_instance_id,
                course_id=next_id,
                name=name,
                duration=duration,
                price_per_participant=price,
                max_participants=max_participants
            )

            # Set participants
            self.db_manager.set_course_participants(
                instance_id=self.current_instance_id,
                course_id=next_id,
                current_participants=current_participants
            )

            self.show_info("Success", f"Added course '{name}'")
            self.clear_course_form()
            self.refresh_courses()

            logger.info(f"Added course {next_id}: {name}")

        except ValueError:
            self.show_error("Error", "Invalid numeric values")
        except Exception as e:
            self.show_error("Error", f"Failed to add course: {e}")
            logger.error(f"Failed to add course: {e}")

    def update_course(self):
        """Update selected course."""
        if not self.selected_course_id:
            self.show_warning("Warning", "Please select a course to update")
            return

        try:
            # Validate data
            name = self.course_name_var.get().strip()
            duration_str = self.course_duration_var.get().strip()
            price_str = self.course_price_var.get().strip()
            max_participants_str = self.course_max_participants_var.get().strip()
            current_participants_str = self.course_current_participants_var.get().strip()

            if not all([name, duration_str, price_str, max_participants_str]):
                self.show_warning("Warning", "Name, duration, price, and max participants are required")
                return

            duration = int(duration_str)
            if duration <= 0:
                self.show_warning("Warning", "Duration must be positive")
                return

            price = float(price_str)
            if price <= 0:
                self.show_warning("Warning", "Price must be positive")
                return

            max_participants = int(max_participants_str)
            if max_participants <= 0:
                self.show_warning("Warning", "Max participants must be positive")
                return

            current_participants = int(current_participants_str) if current_participants_str else max_participants
            if current_participants < 0 or current_participants > max_participants:
                self.show_warning("Warning", "Current participants must be between 0 and max participants")
                return

            # Update course
            success = self.db_manager.update_course(
                instance_id=self.current_instance_id,
                course_id=self.selected_course_id,
                name=name,
                duration=duration,
                price_per_participant=price,
                max_participants=max_participants
            )

            # Update participants
            self.db_manager.set_course_participants(
                instance_id=self.current_instance_id,
                course_id=self.selected_course_id,
                current_participants=current_participants
            )

            if success:
                self.show_info("Success", f"Updated course '{name}'")
                self.clear_course_form()
                self.refresh_courses()
                logger.info(f"Updated course {self.selected_course_id}: {name}")
            else:
                self.show_error("Error", "Failed to update course")

        except ValueError:
            self.show_error("Error", "Invalid numeric values")
        except Exception as e:
            self.show_error("Error", f"Failed to update course: {e}")
            logger.error(f"Failed to update course {self.selected_course_id}: {e}")

    def delete_course(self):
        """Delete selected course."""
        if not self.selected_course_id:
            self.show_warning("Warning", "Please select a course to delete")
            return

        # Get course name for confirmation
        try:
            courses = self.db_manager.get_courses(self.current_instance_id)
            course = next((c for c in courses if c['course_id'] == self.selected_course_id), None)
            course_name = course['name'] if course else f"ID {self.selected_course_id}"
        except:
            course_name = f"ID {self.selected_course_id}"

        if self.show_confirmation("Confirm Delete", f"Are you sure you want to delete course '{course_name}'?"):
            try:
                # Delete course (will cascade to participants and compatibility)
                with self.db_manager.get_connection() as conn:
                    conn.execute(
                        "DELETE FROM courses WHERE instance_id = ? AND course_id = ?",
                        (self.current_instance_id, self.selected_course_id)
                    )
                    conn.commit()

                self.show_info("Success", f"Deleted course '{course_name}'")
                self.clear_course_form()
                self.refresh_courses()
                logger.info(f"Deleted course {self.selected_course_id}: {course_name}")

            except Exception as e:
                self.show_error("Error", f"Failed to delete course: {e}")
                logger.error(f"Failed to delete course {self.selected_course_id}: {e}")

    def clear_course_form(self):
        """Clear course form."""
        self.selected_course_id = None
        self.course_name_var.set('')
        self.course_duration_var.set('')
        self.course_price_var.set('')
        self.course_max_participants_var.set('')
        self.course_current_participants_var.set('')

    # ============================================================================
    # ALGORITHM EXECUTION METHODS
    # ============================================================================

    def run_algorithm(self):
        """Run selected algorithm."""
        if not self.check_instance_selected():
            return

        algorithm_name = self.algorithm_var.get()
        variant = self.variant_var.get()

        if not algorithm_name:
            self.show_warning("Warning", "Please select an algorithm")
            return

        try:
            # Disable interface
            self.start_algorithm_execution()

            # Start algorithm in background thread
            self.algorithm_thread = threading.Thread(
                target=self.execute_algorithm_thread,
                args=(algorithm_name, variant),
                daemon=True
            )
            self.algorithm_thread.start()

            logger.info(f"Started algorithm {algorithm_name} with variant {variant}")

        except Exception as e:
            self.show_error("Error", f"Failed to start algorithm: {e}")
            self.stop_algorithm_execution()
            logger.error(f"Failed to start algorithm: {e}")

    def execute_algorithm_thread(self, algorithm_name, variant):
        """Execute algorithm in background thread."""
        try:
            # Update progress
            self.progress_queue.put("Loading problem from database...")

            # Load problem from database
            problem = self.adapter.create_problem_from_db(self.current_instance_id)

            self.progress_queue.put("Initializing algorithm...")

            # Import and create algorithm
            algorithm = self.create_algorithm_instance(algorithm_name, variant, problem)

            if not algorithm:
                self.progress_queue.put("ERROR: Algorithm not available")
                self.root.after(100, self.stop_algorithm_execution)
                return

            self.progress_queue.put("Running algorithm...")

            # Run algorithm
            solution = algorithm.run()

            self.progress_queue.put("Processing results...")

            # Process results
            if algorithm.best_solution:
                x, y, u = algorithm.best_solution.to_arrays()
                objective = problem.calculate_objective(x, y, u)
                is_feasible, violations = problem.validate_solution(x, y, u)
                assignments = algorithm.best_solution.get_all_assignments()

                result_data = {
                    'success': True,
                    'execution_time': algorithm.stats.execution_time,
                    'objective_value': objective,
                    'is_feasible': is_feasible,
                    'num_violations': len(violations) if violations else 0,
                    'scheduled_courses': len(assignments),
                    'total_courses': problem.N,
                    'utilization': algorithm.best_solution.calculate_utilization(),
                    'detailed_breakdown': problem.calculate_detailed_objective(x, y, u)
                }

                # Save to database
                run_id = self.adapter.save_algorithm_result(
                    instance_id=self.current_instance_id,
                    algorithm_name=algorithm_name,
                    algorithm_variant=variant,
                    parameters={'variant': variant},
                    result=result_data,
                    solution_assignments=assignments
                )

                # Schedule GUI updates
                self.root.after(100,
                                lambda: self.algorithm_completed(result_data, assignments, algorithm_name, variant,
                                                                 run_id))

            else:
                self.root.after(100, lambda: self.algorithm_failed("Algorithm did not produce a solution"))

        except ImportError as e:
            error_msg = f"Could not import algorithm {algorithm_name}: {e}"
            self.root.after(100, lambda: self.algorithm_failed(error_msg))
        except Exception as e:
            error_msg = f"Algorithm execution failed: {e}"
            self.root.after(100, lambda: self.algorithm_failed(error_msg))
            logger.error(f"Algorithm execution failed: {e}", exc_info=True)

    def create_algorithm_instance(self, algorithm_name, variant, problem):
        """Create algorithm instance based on name and variant (FIXED)."""
        try:
            # Add algorithm path to Python path
            algorithms_path = str(Path(__file__).parent.parent / "src" / "algorithms")
            if algorithms_path not in sys.path:
                sys.path.append(algorithms_path)

            # Get parameters for this variant
            variant_params = self.algorithm_parameters.get(algorithm_name, {}).get(variant, {})
            if not variant_params:
                variant_params = self.get_algorithm_variant_parameters(algorithm_name, variant)

            # Ensure variant_params is a dictionary
            if variant_params is None:
                variant_params = {}
            elif not isinstance(variant_params, dict):
                variant_params = {}

            # Remove 'name' from variant_params to avoid conflict
            clean_params = {k: v for k, v in variant_params.items() if k != 'name'}

            # Create algorithm name
            algorithm_display_name = f"{algorithm_name}_{variant}"

            if algorithm_name == "GreedyAlgorithm":
                from greedy import GreedyAlgorithm
                return GreedyAlgorithm(
                    problem,
                    name=algorithm_display_name,
                    verbose=False,
                    **clean_params
                )

            elif algorithm_name == "GeneticAlgorithm":
                from genetic import GeneticAlgorithm
                return GeneticAlgorithm(
                    problem,
                    name=algorithm_display_name,
                    verbose=False,
                    **clean_params
                )

            elif algorithm_name == "SimulatedAnnealingAlgorithm":
                from simulated_annealing import SimulatedAnnealingAlgorithm
                return SimulatedAnnealingAlgorithm(
                    problem,
                    name=algorithm_display_name,
                    verbose=False,
                    **clean_params
                )

            elif algorithm_name == "TabuSearchAlgorithm":
                from tabu_search import TabuSearchAlgorithm
                return TabuSearchAlgorithm(
                    problem,
                    name=algorithm_display_name,
                    verbose=False,
                    **clean_params
                )

            elif algorithm_name == "VariableNeighborhoodSearch":
                from vns import VariableNeighborhoodSearch
                return VariableNeighborhoodSearch(
                    problem,
                    name=algorithm_display_name,
                    verbose=False,
                    **clean_params
                )
            else:
                logger.error(f"Unknown algorithm: {algorithm_name}")
                return None

        except ImportError as e:
            logger.error(f"Failed to import {algorithm_name}: {e}")
            return None
        except TypeError as e:
            logger.error(f"Failed to create {algorithm_name} due to parameter error: {e}")
            logger.error(f"Parameters used: {clean_params}")
            return None
        except Exception as e:
            logger.error(f"Failed to create {algorithm_name}: {e}")
            return None

    def algorithm_completed(self, result_data, assignments, algorithm_name, variant, run_id):
        """Handle algorithm completion."""
        try:
            # Display results
            self.display_algorithm_result(result_data, assignments, algorithm_name, variant)

            # Refresh results tab
            self.refresh_results()

            # Show success message
            objective = result_data['objective_value']
            feasible = "feasible" if result_data['is_feasible'] else "infeasible"
            time_taken = result_data['execution_time']

            self.show_info("Algorithm Completed",
                           f"Algorithm: {algorithm_name}\n"
                           f"Objective: {objective:.2f} ({feasible})\n"
                           f"Time: {time_taken:.3f} seconds\n"
                           f"Run ID: {run_id}")

            logger.info(f"Algorithm {algorithm_name} completed successfully - Run ID: {run_id}")

        except Exception as e:
            self.show_error("Error", f"Failed to process algorithm results: {e}")
            logger.error(f"Failed to process algorithm results: {e}")
        finally:
            self.stop_algorithm_execution()

    def algorithm_failed(self, error_message):
        """Handle algorithm failure."""
        self.show_error("Algorithm Failed", error_message)
        logger.error(f"Algorithm failed: {error_message}")
        self.stop_algorithm_execution()

    def start_algorithm_execution(self):
        """Prepare GUI for algorithm execution."""
        self.run_button.config(state="disabled")
        self.progress_bar.start()
        self.progress_var.set("Starting algorithm...")

        # Disable instance selection and data modification during execution
        self.instance_combo.config(state="disabled")

        # Switch to algorithm tab if not already there
        self.notebook.select(1)  # Algorithm tab is index 1

    def stop_algorithm_execution(self):
        """Restore GUI after algorithm execution."""
        self.run_button.config(state="normal")
        self.progress_bar.stop()
        self.progress_var.set("Ready")
        self.instance_combo.config(state="readonly")

    def display_algorithm_result(self, result_data, assignments, algorithm_name, variant):
        """Display algorithm result in the preview area."""
        self.result_text.delete(1.0, tk.END)

        result_text = f"Algorithm: {algorithm_name} - {variant}\n"
        result_text += "=" * 60 + "\n\n"

        # Basic results
        result_text += f"Objective Value: {result_data['objective_value']:.2f}\n"
        result_text += f"Feasible: {'Yes' if result_data['is_feasible'] else 'No'}\n"
        result_text += f"Violations: {result_data['num_violations']}\n"
        result_text += f"Execution Time: {result_data['execution_time']:.3f} seconds\n"
        result_text += f"Scheduled Courses: {result_data['scheduled_courses']}/{result_data['total_courses']}\n\n"

        # Utilization
        util = result_data['utilization']
        result_text += f"Utilization:\n"
        result_text += f"  Room: {util.get('room_utilization', 0):.1%}\n"
        result_text += f"  Instructor: {util.get('instructor_utilization', 0):.1%}\n"
        result_text += f"  Courses Scheduled: {util.get('courses_scheduled', 0):.1%}\n\n"

        # Cost breakdown
        if 'detailed_breakdown' in result_data and result_data['detailed_breakdown']:
            breakdown = result_data['detailed_breakdown']
            result_text += f"Cost Breakdown:\n"
            result_text += f"  Revenue: ${breakdown.get('zysk', 0):.2f}\n"
            result_text += f"  Room Usage: ${breakdown.get('koszt_uzytkowania', 0):.2f}\n"
            result_text += f"  Room Idle: ${breakdown.get('koszt_niewykorzystania', 0):.2f}\n"
            result_text += f"  Instructors: ${breakdown.get('koszt_instruktorow', 0):.2f}\n\n"

        # Assignments
        if assignments:
            result_text += f"Course Assignments ({len(assignments)} total):\n"
            result_text += "-" * 40 + "\n"

            # Sort assignments by start time for better readability
            sorted_assignments = sorted(assignments, key=lambda x: (x['start_time'], x['room_id']))

            for i, assignment in enumerate(sorted_assignments):
                result_text += f"{i + 1:2d}. Course {assignment['course_id']}: "
                result_text += f"Room {assignment['room_id']}, "
                result_text += f"Instructor {assignment['instructor_id']}, "
                result_text += f"Time {assignment['start_time']}-{assignment['end_time']}, "
                result_text += f"{assignment['participants']} participants\n"

                # Add line break every 10 assignments for readability
                if (i + 1) % 10 == 0 and i < len(sorted_assignments) - 1:
                    result_text += "\n"

        else:
            result_text += "No course assignments found.\n"

        # Additional info
        result_text += "\n" + "=" * 60 + "\n"
        result_text += f"Algorithm completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        result_text += f"Instance: {self.current_instance_name} (ID: {self.current_instance_id})\n"

        self.result_text.insert(1.0, result_text)

        # Auto-scroll to top
        self.result_text.see(1.0)

    # ============================================================================
    # RESULTS MANAGEMENT METHODS
    # ============================================================================

    def refresh_results(self):
        """Refresh results list."""
        if not self.current_instance_id:
            return

        try:
            # Clear tree
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)

            # Load results
            results = self.db_manager.get_schedule_runs(self.current_instance_id)

            for result in results:
                # Format values safely
                objective_str = f"{result['objective_value']:.2f}" if result[
                                                                          'objective_value'] is not None else 'N/A'
                feasible_str = 'Yes' if result['is_feasible'] else 'No' if result[
                                                                               'is_feasible'] is not None else 'N/A'
                time_str = f"{result['execution_time_seconds']:.3f}" if result[
                                                                            'execution_time_seconds'] is not None else 'N/A'

                self.results_tree.insert('', tk.END,
                                         text=str(result['run_id']),
                                         values=(
                                             result['algorithm_name'] or '',
                                             result['algorithm_variant'] or '',
                                             objective_str,
                                             feasible_str,
                                             time_str
                                         ))

            logger.info(f"Refreshed {len(results)} algorithm results")

        except Exception as e:
            self.show_error("Error", f"Failed to load results: {e}")
            logger.error(f"Failed to refresh results: {e}")

    def view_schedule(self, event):
        """View schedule grid for selected result."""
        selection = self.results_tree.selection()
        if not selection:
            return

        item = self.results_tree.item(selection[0])
        run_id = int(item['text'])

        try:
            grid_data = self.adapter.get_schedule_as_grid(run_id)
            self.display_schedule_grid(grid_data)

            # Switch to results tab to show the grid
            self.notebook.select(2)  # Results tab is index 2

            logger.info(f"Viewing schedule for run ID {run_id}")

        except Exception as e:
            self.show_error("Error", f"Failed to load schedule: {e}")
            logger.error(f"Failed to load schedule for run {run_id}: {e}")

    def delete_selected_result(self):
        """Delete selected algorithm result."""
        selection = self.results_tree.selection()
        if not selection:
            self.show_warning("Warning", "Please select a result to delete")
            return

        item = self.results_tree.item(selection[0])
        run_id = int(item['text'])
        algorithm_name = item['values'][0]

        if self.show_confirmation("Confirm Delete",
                                  f"Are you sure you want to delete result from '{algorithm_name}' (Run ID: {run_id})?"):
            try:
                success = self.db_manager.delete_schedule_run(run_id)

                if success:
                    self.show_info("Success", f"Deleted result (Run ID: {run_id})")
                    self.refresh_results()

                    # Clear grid if this result was being displayed
                    self.grid_text.delete(1.0, tk.END)

                    logger.info(f"Deleted algorithm result run ID {run_id}")
                else:
                    self.show_error("Error", "Failed to delete result")

            except Exception as e:
                self.show_error("Error", f"Failed to delete result: {e}")
                logger.error(f"Failed to delete result {run_id}: {e}")

    def show_web_visualization(self):
        """Show web visualization for selected algorithm result."""
        selection = self.results_tree.selection()
        if not selection:
            self.show_warning("Warning", "Please select a result to visualize")
            return

        item = self.results_tree.item(selection[0])
        run_id = int(item['text'])
        algorithm_name = item['values'][0]
        algorithm_variant = item['values'][1] if len(item['values']) > 1 else ""

        try:
            # Get schedule data from database
            self.progress_var.set("Preparing visualization...")
            self.root.update()

            # Get grid data
            grid_data = self.adapter.get_schedule_as_grid(run_id)

            # Convert to format expected by HTML visualization
            viz_data = self._convert_to_visualization_format(grid_data)

            # Generate and open HTML file
            self._generate_and_open_html(viz_data, f"{algorithm_name}_{algorithm_variant}_{run_id}")

            self.show_info("Visualization Opened",
                           f"Web visualization for {algorithm_name} opened in browser")

            logger.info(f"Opened web visualization for run ID {run_id}")

        except Exception as e:
            self.show_error("Error", f"Failed to open visualization: {e}")
            logger.error(f"Failed to open visualization for run {run_id}: {e}")
        finally:
            self.progress_var.set("Ready")

    def _convert_to_visualization_format(self, grid_data):
        """Convert database grid data to format expected by HTML visualization."""
        grid = grid_data['grid']
        metadata = grid_data['metadata']

        # Extract course assignments from grid
        course_assignments = []
        processed_courses = set()

        for time_slot, rooms in grid.items():
            for room_id, assignment in rooms.items():
                if assignment['is_start']:  # Only process course starts
                    course_id = assignment['course_id']
                    if course_id not in processed_courses:
                        course_assignments.append({
                            'course_id': course_id,
                            'room_id': str(room_id),
                            'instructor_id': str(assignment['instructor_id']),
                            'start_time': time_slot,
                            'end_time': time_slot + assignment['duration'] - 1,
                            'duration': assignment['duration'],
                            'participants': assignment['participants']
                        })
                        processed_courses.add(course_id)

        # Calculate utilization
        total_slots = metadata['time_horizon'] * len(metadata['room_names'])
        used_slots = sum(len(rooms) for rooms in grid.values())
        room_utilization = used_slots / total_slots if total_slots > 0 else 0

        # Build visualization data structure
        viz_data = {
            'solution': {
                'objective_value': metadata.get('objective_value', 0),
                'course_assignments': course_assignments,
                'utilization': {
                    'room_utilization': room_utilization
                }
            },
            'problem_info': {
                'num_courses': metadata.get('total_courses', len(course_assignments)),
                'num_rooms': len(metadata['room_names']),
                'num_instructors': len(metadata['instructor_names']),
                'time_horizon': metadata['time_horizon']
            },
            'metadata': {
                'run_id': metadata['run_id'],
                'algorithm_name': metadata['algorithm_name'],
                'algorithm_variant': metadata.get('algorithm_variant', ''),
                'instance_name': metadata['instance_name'],
                'is_feasible': metadata.get('is_feasible', False),
                'execution_time': metadata.get('execution_time', 0)
            }
        }

        return viz_data

    def _generate_and_open_html(self, viz_data, filename_prefix):
        """Generate HTML file with visualization and open in browser."""
        import tempfile
        import webbrowser
        import json
        import os

        # Create temporary HTML file
        temp_dir = tempfile.gettempdir()
        filename = f"schedule_visualization_{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        html_path = os.path.join(temp_dir, filename)

        # Get HTML template
        html_template = self._get_html_template()

        # Inject data into template
        data_json = json.dumps(viz_data, indent=2, ensure_ascii=False)
        html_content = html_template.replace('{{DATA_PLACEHOLDER}}', data_json)

        # Write HTML file
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Open in browser
        webbrowser.open('file://' + html_path.replace('\\', '/'))

        logger.info(f"Generated visualization HTML: {html_path}")

    def _get_html_template(self):
        """Get HTML template for visualization."""
        # This is the HTML template based on your wiz_basic2.html
        # with modifications to accept dynamic data
        return '''<!DOCTYPE html>
    <html lang="pl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Harmonogram Zajęć - Wizualizacja</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }

            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                padding: 30px;
            }

            .header {
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 3px solid #f0f0f0;
            }

            .header h1 {
                color: #2c3e50;
                font-size: 2.5em;
                margin-bottom: 10px;
                background: linear-gradient(45deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .algorithm-info {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                text-align: center;
            }

            .algorithm-info h2 {
                color: #495057;
                margin-bottom: 5px;
            }

            .algorithm-info p {
                color: #6c757d;
                margin: 2px 0;
            }

            .stats {
                display: flex;
                justify-content: space-around;
                margin-bottom: 30px;
                background: #f8f9fa;
                padding: 20px;
                border-radius: 15px;
                flex-wrap: wrap;
                gap: 10px;
            }

            .stat-item {
                text-align: center;
                padding: 10px;
                min-width: 120px;
            }

            .stat-value {
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
            }

            .stat-label {
                color: #666;
                font-size: 0.9em;
                margin-top: 5px;
            }

            .schedule-container {
                overflow-x: auto;
                background: #f8f9fa;
                border-radius: 15px;
                padding: 20px;
                margin: 20px 0;
            }

            .schedule-grid {
                display: grid;
                gap: 2px;
                background: #e9ecef;
                border-radius: 10px;
                padding: 15px;
                min-width: fit-content;
            }

            .time-header {
                background: #495057;
                color: white;
                padding: 15px 10px;
                text-align: center;
                font-weight: bold;
                border-radius: 8px;
            }

            .time-slot {
                background: #6c757d;
                color: white;
                padding: 12px 8px;
                text-align: center;
                font-weight: bold;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 60px;
            }

            .instructor-header {
                background: #28a745;
                color: black;
                padding: 15px;
                text-align: center;
                font-weight: bold;
                border-radius: 8px;
                min-width: 120px;
            }

            .course-block {
                color: #2c3e50;
                padding: 12px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                cursor: pointer;
                position: relative;
                overflow: hidden;
                min-height: 60px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                min-width: 120px;
            }

            .course-block:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            }

            .course-id {
                font-size: 1.1em;
                font-weight: bold;
                margin-bottom: 4px;
            }

            .course-details {
                font-size: 0.8em;
                opacity: 0.8;
                line-height: 1.2;
            }

            .empty-slot {
                background: #f8f9fa;
                border: 2px dashed #dee2e6;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #6c757d;
                font-style: italic;
                min-height: 60px;
                min-width: 120px;
            }

            .legend {
                display: flex;
                justify-content: center;
                gap: 20px;
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 15px;
                flex-wrap: wrap;
            }

            .legend-item {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 8px 16px;
                background: white;
                border-radius: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }

            .legend-color {
                width: 20px;
                height: 20px;
                border-radius: 4px;
            }

            .tooltip {
                position: absolute;
                background: #2c3e50;
                color: white;
                padding: 12px 16px;
                border-radius: 8px;
                font-size: 0.9em;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                z-index: 1000;
                opacity: 0;
                transform: translateY(10px);
                transition: all 0.3s ease;
                pointer-events: none;
                max-width: 280px;
                line-height: 1.4;
            }

            .tooltip.show {
                opacity: 1;
                transform: translateY(0);
            }

            .tooltip::after {
                content: '';
                position: absolute;
                top: 100%;
                left: 50%;
                transform: translateX(-50%);
                border: 6px solid transparent;
                border-top-color: #2c3e50;
            }

            .instructor-colors {
                --color-1: linear-gradient(135deg, #ff6b6b, #ee5a24);
                --color-2: linear-gradient(135deg, #4ecdc4, #44a08d);
                --color-3: linear-gradient(135deg, #a8e6cf, #7fcdcd);
                --color-4: linear-gradient(135deg, #ffd93d, #ff9ff3);
                --color-5: linear-gradient(135deg, #74b9ff, #0984e3);
                --color-6: linear-gradient(135deg, #fd79a8, #e84393);
                --color-7: linear-gradient(135deg, #fdcb6e, #e17055);
                --color-8: linear-gradient(135deg, #6c5ce7, #a29bfe);
                --color-9: linear-gradient(135deg, #00b894, #00cec9);
                --color-10: linear-gradient(135deg, #e17055, #d63031);
            }

            @media (max-width: 1024px) {
                .schedule-container {
                    padding: 10px;
                }
                .course-block, .empty-slot, .instructor-header {
                    min-width: 100px;
                    font-size: 0.9em;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📅 Harmonogram Zajęć</h1>
                <p>Wizualizacja wyników optymalizacji</p>
            </div>

            <div class="algorithm-info" id="algorithmInfo">
                <!-- Informacje o algorytmie -->
            </div>

            <div class="stats" id="statsContainer">
                <div class="stat-item">
                    <div class="stat-value" id="coursesCount">-</div>
                    <div class="stat-label">Kursy</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="roomsCount">-</div>
                    <div class="stat-label">Sale</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="instructorsCount">-</div>
                    <div class="stat-label">Instruktorzy</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="objectiveValue">-</div>
                    <div class="stat-label">Funkcja celu</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="utilizationValue">-</div>
                    <div class="stat-label">Wykorzystanie</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="timeRange">-</div>
                    <div class="stat-label">Przedział czasowy</div>
                </div>
            </div>

            <div class="schedule-container">
                <div class="schedule-grid" id="scheduleGrid">
                    <!-- Generowane dynamicznie -->
                </div>
            </div>

            <div class="legend" id="legendContainer">
                <!-- Generowane dynamicznie -->
            </div>
        </div>

        <div class="tooltip" id="tooltip"></div>

        <script>
            // Data will be injected here
            const currentData = {{DATA_PLACEHOLDER}};

            function analyzeData(data) {
            const courses = data.solution.course_assignments || [];
            
            if (courses.length === 0) {
                console.warn('Brak danych o kursach');
                return {
                    courses: [],
                    instructors: [],
                    rooms: [],
                    timeRange: { min: 1, max: 1 },
                    stats: {
                        numCourses: 0,
                        numInstructors: 0,
                        numRooms: 0,
                        objectiveValue: 0,
                        utilization: 0
                    }
                };
            }

            const instructors = [...new Set(courses.map(c => c.instructor_id))].sort();
            const rooms = [...new Set(courses.map(c => c.room_id))].sort();
            const times = courses.flatMap(c => {
                const times = [];
                for (let t = c.start_time; t <= c.end_time; t++) {
                    times.push(t);
                }
                return times;
            });
            const minTime = Math.min(...times);
            const maxTime = Math.max(...times);

            return {
                courses,
                instructors,
                rooms,
                timeRange: { min: minTime, max: maxTime },
                stats: {
                    numCourses: courses.length,
                    numInstructors: instructors.length,
                    numRooms: rooms.length,
                    objectiveValue: data.solution.objective_value || 0,
                    utilization: data.solution.utilization?.room_utilization || 0
                }
            };
        }

        function generateInstructorColors(count) {
            const colors = [];
            for (let i = 1; i <= Math.min(count, 10); i++) {
                colors.push(`var(--color-${i})`);
            }

            for (let i = 11; i <= count; i++) {
                const hue = ((i - 11) * 137.508) % 360;
                colors.push(`hsl(${hue}, 70%, 60%)`);
            }

            return colors;
        }

        function updateAlgorithmInfo(data) {
            const algorithmInfo = document.getElementById('algorithmInfo');
            const metadata = data.metadata || {};
            
            algorithmInfo.innerHTML = `
                <h2>🤖 ${metadata.algorithm_name || 'Unknown'} ${metadata.algorithm_variant ? '- ' + metadata.algorithm_variant : ''}</h2>
                <p>📊 Instancja: ${metadata.instance_name || 'Unknown'}</p>
                <p>🎯 Status: ${metadata.is_feasible ? '✅ Wykonalne rozwiązanie' : '❌ Niewykonalne'}</p>
                <p>⏱️ Czas wykonania: ${metadata.execution_time ? metadata.execution_time.toFixed(3) + 's' : 'N/A'}</p>
                <p>🔢 ID uruchomienia: ${metadata.run_id || 'N/A'}</p>
            `;
        }

        function updateStats(analysis) {
            document.getElementById('coursesCount').textContent = analysis.stats.numCourses;
            document.getElementById('roomsCount').textContent = analysis.stats.numRooms;
            document.getElementById('instructorsCount').textContent = analysis.stats.numInstructors;
            document.getElementById('objectiveValue').textContent = Math.round(analysis.stats.objectiveValue);
            document.getElementById('utilizationValue').textContent = `${(analysis.stats.utilization * 100).toFixed(1)}%`;
            document.getElementById('timeRange').textContent = `${analysis.timeRange.min}-${analysis.timeRange.max}`;
        }

        function generateSchedule(analysis) {
            const { courses, instructors, timeRange } = analysis;
            const grid = document.getElementById('scheduleGrid');
            const colors = generateInstructorColors(instructors.length);

            grid.innerHTML = '';

            const timeSlots = [];
            for (let t = timeRange.min; t <= timeRange.max; t++) {
                timeSlots.push(t);
            }

            grid.style.gridTemplateColumns = `80px repeat(${instructors.length}, 1fr)`;
            grid.style.gridTemplateRows = `auto repeat(${timeSlots.length}, 1fr)`;

            const timeHeader = document.createElement('div');
            timeHeader.className = 'time-header';
            timeHeader.textContent = 'Czas';
            grid.appendChild(timeHeader);

            instructors.forEach((instructorId, index) => {
                const instructorHeader = document.createElement('div');
                instructorHeader.className = 'instructor-header';
                instructorHeader.textContent = `Instruktor ${instructorId}`;
                instructorHeader.style.background = colors[index];
                grid.appendChild(instructorHeader);
            });

            timeSlots.forEach(timeSlot => {
                const timeSlotDiv = document.createElement('div');
                timeSlotDiv.className = 'time-slot';
                timeSlotDiv.textContent = timeSlot;
                grid.appendChild(timeSlotDiv);

                instructors.forEach((instructorId, instructorIndex) => {
                    const course = findCourseInSlot(instructorId, timeSlot, courses);
                    const cell = document.createElement('div');

                    if (course) {
                        cell.className = 'course-block';
                        cell.style.background = colors[instructorIndex];
                        cell.innerHTML = `
                            <div class="course-id">Kurs ${course.course_id}</div>
                            <div class="course-details">
                                🏢 Sala ${course.room_id}<br>
                                👥 ${course.participants}<br>
                                ⏱️ ${course.duration}h
                            </div>
                        `;

                        cell.addEventListener('mouseenter', (e) => showTooltip(e, course));
                        cell.addEventListener('mouseleave', hideTooltip);
                        cell.addEventListener('mousemove', updateTooltipPosition);
                    } else {
                        cell.className = 'empty-slot';
                        cell.textContent = '—';
                    }

                    grid.appendChild(cell);
                });
            });

            generateLegend(instructors, colors);
        }

        function generateLegend(instructors, colors) {
            const legend = document.getElementById('legendContainer');
            legend.innerHTML = '';

            instructors.forEach((instructorId, index) => {
                const legendItem = document.createElement('div');
                legendItem.className = 'legend-item';

                const colorBox = document.createElement('div');
                colorBox.className = 'legend-color';
                colorBox.style.background = colors[index];

                const label = document.createElement('span');
                label.textContent = `Instruktor ${instructorId}`;

                legendItem.appendChild(colorBox);
                legendItem.appendChild(label);
                legend.appendChild(legendItem);
            });
        }

        function findCourseInSlot(instructorId, timeSlot, courses) {
            return courses.find(course =>
                course.instructor_id == instructorId &&
                timeSlot >= course.start_time &&
                timeSlot <= course.end_time
            );
        }

        function showTooltip(event, course) {
            const tooltip = document.getElementById('tooltip');
            tooltip.innerHTML = `
                <strong>📚 Kurs ${course.course_id}</strong><br>
                🏢 Sala: ${course.room_id}<br>
                👨‍🏫 Instruktor: ${course.instructor_id}<br>
                ⏰ Czas: ${course.start_time} - ${course.end_time}<br>
                📏 Czas trwania: ${course.duration}h<br>
                👥 Uczestnicy: ${course.participants} osób
            `;
            tooltip.classList.add('show');
            updateTooltipPosition(event);
        }

        function hideTooltip() {
            const tooltip = document.getElementById('tooltip');
            tooltip.classList.remove('show');
        }

        function updateTooltipPosition(event) {
            const tooltip = document.getElementById('tooltip');
            const rect = tooltip.getBoundingClientRect();
            const x = event.clientX;
            const y = event.clientY;

            let left = x + 10;
            let top = y - rect.height - 10;

            if (left + rect.width > window.innerWidth) {
                left = x - rect.width - 10;
            }

            if (top < 0) {
                top = y + 10;
            }

            tooltip.style.left = left + 'px';
            tooltip.style.top = top + 'px';
        }

        // Inicjalizacja
        document.addEventListener('DOMContentLoaded', () => {
            try {
                updateAlgorithmInfo(currentData);
                const analysis = analyzeData(currentData);
                updateStats(analysis);
                generateSchedule(analysis);
                
                // Animacja ładowania
                setTimeout(() => {
                    const blocks = document.querySelectorAll('.course-block');
                    blocks.forEach((block, index) => {
                        block.style.opacity = '0';
                        block.style.transform = 'translateY(20px)';
                        setTimeout(() => {
                            block.style.transition = 'all 0.5s ease';
                            block.style.opacity = '1';
                            block.style.transform = 'translateY(0)';
                        }, index * 50);
                    });
                }, 100);
            } catch (error) {
                console.error('Błąd inicjalizacji wizualizacji:', error);
                document.body.innerHTML = `
                    <div style="padding: 50px; text-align: center; color: red;">
                        <h1>Błąd ładowania wizualizacji</h1>
                        <p>${error.message}</p>
                    </div>
                `;
            }
        });
        </script>
    </body>
    </html>'''

    def display_schedule_grid(self, grid_data):
        """Display schedule grid."""
        self.grid_text.delete(1.0, tk.END)

        grid = grid_data['grid']
        metadata = grid_data['metadata']

        # Header
        header_text = f"Schedule for Run ID {metadata['run_id']}\n"
        header_text += f"Algorithm: {metadata['algorithm_name']}"
        if metadata.get('algorithm_variant'):
            header_text += f" - {metadata['algorithm_variant']}"
        header_text += "\n"
        header_text += f"Instance: {metadata['instance_name']}\n"
        header_text += f"Objective: {metadata['objective_value']:.2f}\n"
        header_text += f"Feasible: {'Yes' if metadata['is_feasible'] else 'No'}\n"
        header_text += f"Execution Time: {metadata.get('execution_time', 0):.3f}s\n"
        header_text += f"Scheduled: {metadata.get('scheduled_courses', 0)}/{metadata.get('total_courses', 0)} courses\n"
        header_text += "=" * 80 + "\n\n"

        # Build grid table
        room_names = metadata['room_names']
        time_horizon = metadata['time_horizon']

        if not room_names:
            self.grid_text.insert(1.0, header_text + "No rooms found in this instance.\n")
            return

        # Create column headers
        room_ids = sorted(room_names.keys())
        col_width = 20

        grid_text = f"{'Time':<6}"
        for room_id in room_ids:
            room_name = room_names[room_id][:col_width - 1]  # Truncate if too long
            grid_text += f"{room_name:<{col_width}}"
        grid_text += "\n"

        grid_text += "-" * (6 + col_width * len(room_ids)) + "\n"

        # Add time slots
        for t in range(1, time_horizon + 1):
            grid_text += f"{t:<6}"

            for room_id in room_ids:
                if t in grid and room_id in grid[t]:
                    assignment = grid[t][room_id]
                    course_name = assignment['course_name'][:12]  # Truncate course name
                    instructor_name = assignment['instructor_name'][:6]  # Truncate instructor

                    if assignment['is_start']:
                        cell_text = f"{course_name}({instructor_name})"
                    else:
                        cell_text = f"↓ {course_name}"

                    cell_text = cell_text[:col_width - 1]  # Ensure it fits
                    grid_text += f"{cell_text:<{col_width}}"
                else:
                    grid_text += f"{'---':<{col_width}}"

            grid_text += "\n"

        # Add legend
        grid_text += "\n" + "=" * 80 + "\n"
        grid_text += "Legend:\n"
        grid_text += "  Course(Instructor) = Course start with instructor\n"
        grid_text += "  ↓ Course = Course continuation\n"
        grid_text += "  --- = Empty slot\n\n"

        # Add course details
        if grid:
            grid_text += "Course Details:\n"
            grid_text += "-" * 40 + "\n"

            # Collect unique courses from the grid
            courses_in_schedule = {}
            for time_slot, rooms in grid.items():
                for room_id, assignment in rooms.items():
                    if assignment['is_start']:  # Only count course starts
                        course_id = assignment['course_id']
                        if course_id not in courses_in_schedule:
                            courses_in_schedule[course_id] = {
                                'name': assignment['course_name'],
                                'instructor': assignment['instructor_name'],
                                'room': room_names[room_id],
                                'start_time': time_slot,
                                'participants': assignment['participants'],
                                'duration': assignment['duration']
                            }

            # Sort by start time
            sorted_courses = sorted(courses_in_schedule.items(), key=lambda x: x[1]['start_time'])

            for course_id, details in sorted_courses:
                grid_text += f"Course {course_id}: {details['name']}\n"
                grid_text += f"  Room: {details['room']}\n"
                grid_text += f"  Instructor: {details['instructor']}\n"
                grid_text += f"  Time: {details['start_time']}-{details['start_time'] + details['duration'] - 1}\n"
                grid_text += f"  Participants: {details['participants']}\n\n"

        # Display full text
        full_text = header_text + grid_text
        self.grid_text.insert(1.0, full_text)

        # Auto-scroll to top
        self.grid_text.see(1.0)

        # ============================================================================
        # UTILITY METHODS
        # ============================================================================

    def check_instance_selected(self):
        """Check if an instance is selected."""
        if not self.current_instance_id:
            self.show_warning("Warning", "Please select an instance first")
            return False
        return True

    def validate_time_slots(self, slots_string):
        """Validate time slots format."""
        if not slots_string.strip():
            return True  # Empty is valid (means all slots)

        try:
            slots = [int(x.strip()) for x in slots_string.split(';') if x.strip()]
            return all(slot > 0 for slot in slots)  # All slots must be positive
        except ValueError:
            return False

    def show_error(self, title, message):
        """Show error dialog."""
        messagebox.showerror(title, message)

    def show_warning(self, title, message):
        """Show warning dialog."""
        messagebox.showwarning(title, message)

    def show_info(self, title, message):
        """Show info dialog."""
        messagebox.showinfo(title, message)

    def show_confirmation(self, title, message):
        """Show confirmation dialog."""
        return messagebox.askyesno(title, message)

# KONIEC KLASY CourseSchedulingGUI

# ============================================================================
# DIALOG CLASSES
# ============================================================================

class NewInstanceDialog:
    """Dialog for creating new instances."""

    def __init__(self, parent, title="Create New Instance"):
        self.result = None

        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("450x350")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.resizable(False, False)

        # Center dialog
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 100, parent.winfo_rooty() + 100))

        self.create_widgets()

    def create_widgets(self):
        """Create dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Create New Instance",
                                font=('TkDefaultFont', 12, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Form fields
        ttk.Label(main_frame, text="Instance Name:").grid(row=1, column=0, sticky=tk.W, pady=8)
        self.name_var = tk.StringVar()
        name_entry = ttk.Entry(main_frame, textvariable=self.name_var, width=30, font=('TkDefaultFont', 10))
        name_entry.grid(row=1, column=1, pady=8, padx=(10, 0))
        name_entry.focus_set()

        ttk.Label(main_frame, text="Description:").grid(row=2, column=0, sticky=tk.W, pady=8)
        self.description_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.description_var, width=30, font=('TkDefaultFont', 10)).grid(row=2,
                                                                                                            column=1,
                                                                                                            pady=8,
                                                                                                            padx=(
                                                                                                            10, 0))

        ttk.Label(main_frame, text="Time Horizon:").grid(row=3, column=0, sticky=tk.W, pady=8)
        self.time_horizon_var = tk.StringVar(value="20")
        ttk.Entry(main_frame, textvariable=self.time_horizon_var, width=30, font=('TkDefaultFont', 10)).grid(row=3,
                                                                                                             column=1,
                                                                                                             pady=8,
                                                                                                             padx=(
                                                                                                             10, 0))

        ttk.Label(main_frame, text="Difficulty:").grid(row=4, column=0, sticky=tk.W, pady=8)
        self.difficulty_var = tk.StringVar(value="medium")
        difficulty_combo = ttk.Combobox(main_frame, textvariable=self.difficulty_var,
                                        values=["easy", "medium", "hard"], state="readonly", width=27)
        difficulty_combo.grid(row=4, column=1, pady=8, padx=(10, 0))

        # Help text
        help_frame = ttk.LabelFrame(main_frame, text="Help", padding="10")
        help_frame.grid(row=5, column=0, columnspan=2, pady=20, sticky=(tk.W, tk.E))

        help_text = ("• Instance Name: Unique identifier for your scheduling problem\n"
                     "• Description: Optional details about the instance\n"
                     "• Time Horizon: Number of time slots available (typically 15-30)\n"
                     "• Difficulty: Affects default parameters for algorithms")

        ttk.Label(help_frame, text=help_text, justify=tk.LEFT, wraplength=400).pack(anchor=tk.W)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=20)

        create_btn = ttk.Button(button_frame, text="Create Instance", command=self.ok_clicked)
        create_btn.pack(side=tk.LEFT, padx=5)

        cancel_btn = ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked)
        cancel_btn.pack(side=tk.LEFT, padx=5)

        # Bind Enter key to create
        self.dialog.bind('<Return>', lambda e: self.ok_clicked())
        self.dialog.bind('<Escape>', lambda e: self.cancel_clicked())

    def ok_clicked(self):
        """Handle Create button click."""
        name = self.name_var.get().strip()
        description = self.description_var.get().strip()
        time_horizon_str = self.time_horizon_var.get().strip()
        difficulty = self.difficulty_var.get()

        # Validation
        if not name:
            messagebox.showwarning("Validation Error", "Instance name is required")
            return

        if len(name) < 3:
            messagebox.showwarning("Validation Error", "Instance name must be at least 3 characters long")
            return

        try:
            time_horizon = int(time_horizon_str)
            if time_horizon <= 0:
                raise ValueError("Time horizon must be positive")
            if time_horizon > 100:
                raise ValueError("Time horizon too large (max 100)")

            self.result = {
                'name': name,
                'description': description or f"Instance created on {datetime.now().strftime('%Y-%m-%d')}",
                'time_horizon': time_horizon,
                'difficulty': difficulty
            }

            self.dialog.destroy()

        except ValueError as e:
            messagebox.showerror("Validation Error", f"Invalid time horizon: {e}")

    def cancel_clicked(self):
        """Handle Cancel button click."""
        self.dialog.destroy()

# ============================================================================

class AlgorithmParametersDialog:
    """Dialog for configuring algorithm parameters."""

    def __init__(self, parent, algorithm_name, variant_name, current_params):
        self.result = None
        self.algorithm_name = algorithm_name
        self.variant_name = variant_name
        self.current_params = current_params.copy()
        self.param_vars = {}

        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"Configure {algorithm_name} - {variant_name}")
        self.dialog.geometry("500x600")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.resizable(True, True)

        # Center dialog
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))

        self.create_widgets()

    def create_widgets(self):
        """Create dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_text = f"Parameters for {self.algorithm_name} - {self.variant_name}"
        title_label = ttk.Label(main_frame, text=title_text, font=('TkDefaultFont', 12, 'bold'))
        title_label.pack(pady=(0, 20))

        # Scrollable frame for parameters
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Parameters frame
        params_frame = ttk.LabelFrame(scrollable_frame, text="Algorithm Parameters", padding="10")
        params_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create parameter inputs
        self.create_parameter_inputs(params_frame)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))

        ttk.Button(button_frame, text="Apply", command=self.apply_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side=tk.RIGHT, padx=5)

        # Bind Enter key
        self.dialog.bind('<Return>', lambda e: self.apply_clicked())
        self.dialog.bind('<Escape>', lambda e: self.cancel_clicked())

    def create_parameter_inputs(self, parent):
        """Create input widgets for algorithm parameters."""
        row = 0

        # Common parameters
        common_params = ['name', 'max_iterations', 'max_time', 'target_objective', 'verbose']

        # Algorithm-specific parameter definitions
        param_definitions = self.get_parameter_definitions()

        for param_name, param_value in self.current_params.items():
            if param_name == 'name':
                continue  # Skip name parameter

            # Get parameter definition
            param_def = param_definitions.get(param_name, {})
            param_type = param_def.get('type', 'string')
            description = param_def.get('description', param_name.replace('_', ' ').title())

            # Create label
            ttk.Label(parent, text=f"{description}:").grid(row=row, column=0, sticky=tk.W, pady=5, padx=(0, 10))

            # Create input widget based on type
            if param_type == 'boolean':
                var = tk.BooleanVar(value=bool(param_value))
                widget = ttk.Checkbutton(parent, variable=var)
            elif param_type == 'choice':
                var = tk.StringVar(value=str(param_value))
                choices = param_def.get('choices', [str(param_value)])
                widget = ttk.Combobox(parent, textvariable=var, values=choices, state="readonly")
            elif param_type in ['integer', 'float']:
                var = tk.StringVar(value=str(param_value))
                widget = ttk.Entry(parent, textvariable=var, width=20)
            else:  # string
                var = tk.StringVar(value=str(param_value))
                widget = ttk.Entry(parent, textvariable=var, width=30)

            widget.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)

            # Store variable
            self.param_vars[param_name] = {'var': var, 'type': param_type}

            # Add help text if available
            help_text = param_def.get('help', '')
            if help_text:
                help_label = ttk.Label(parent, text=help_text, font=('TkDefaultFont', 8), foreground='gray')
                help_label.grid(row=row, column=2, sticky=tk.W, pady=5, padx=(10, 0))

            row += 1

        # Configure column weights
        parent.columnconfigure(1, weight=1)

    def get_parameter_definitions(self):
        """Get parameter definitions for current algorithm."""
        definitions = {}

        if self.algorithm_name == "TabuSearchAlgorithm":
            definitions = {
                'tabu_tenure': {'type': 'integer', 'description': 'Tabu Tenure', 'help': 'Length of tabu list'},
                'dynamic_tenure': {'type': 'boolean', 'description': 'Dynamic Tenure',
                                   'help': 'Adjust tenure dynamically'},
                'min_tenure': {'type': 'integer', 'description': 'Min Tenure', 'help': 'Minimum tabu tenure'},
                'max_tenure': {'type': 'integer', 'description': 'Max Tenure', 'help': 'Maximum tabu tenure'},
                'neighborhood_strategy': {'type': 'choice', 'description': 'Neighborhood Strategy',
                                          'choices': ['best_improvement', 'first_improvement', 'candidate_list',
                                                      'random_sampling'],
                                          'help': 'How to explore neighborhood'},
                'candidate_list_size': {'type': 'integer', 'description': 'Candidate List Size',
                                        'help': 'Size of candidate list'},
                'aspiration_criteria': {'type': 'boolean', 'description': 'Aspiration Criteria',
                                        'help': 'Use aspiration criteria'},
                'intensification_enabled': {'type': 'boolean', 'description': 'Intensification',
                                            'help': 'Enable intensification'},
                'diversification_enabled': {'type': 'boolean', 'description': 'Diversification',
                                            'help': 'Enable diversification'},
                'intensification_threshold': {'type': 'integer', 'description': 'Intensification Threshold',
                                              'help': 'Iterations before intensification'},
                'diversification_threshold': {'type': 'integer', 'description': 'Diversification Threshold',
                                              'help': 'Iterations before diversification'},
                'diversification_method': {'type': 'choice', 'description': 'Diversification Method',
                                           'choices': ['restart', 'big_perturbation', 'frequency_based',
                                                       'elite_solutions'],
                                           'help': 'How to diversify'},
                'elite_size': {'type': 'integer', 'description': 'Elite Size',
                               'help': 'Number of elite solutions to keep'},
                'max_iterations': {'type': 'integer', 'description': 'Max Iterations',
                                   'help': 'Maximum number of iterations'}
            }

        elif self.algorithm_name == "GeneticAlgorithm":
            definitions = {
                'population_size': {'type': 'integer', 'description': 'Population Size',
                                    'help': 'Number of individuals in population'},
                'max_generations': {'type': 'integer', 'description': 'Max Generations',
                                    'help': 'Maximum number of generations'},
                'crossover_rate': {'type': 'float', 'description': 'Crossover Rate',
                                   'help': 'Probability of crossover (0.0-1.0)'},
                'mutation_rate': {'type': 'float', 'description': 'Mutation Rate',
                                  'help': 'Probability of mutation (0.0-1.0)'},
                'selection_method': {'type': 'choice', 'description': 'Selection Method',
                                     'choices': ['tournament', 'roulette', 'rank', 'elitist'],
                                     'help': 'Parent selection method'},
                'tournament_size': {'type': 'integer', 'description': 'Tournament Size',
                                    'help': 'Size for tournament selection'},
                'elitism_rate': {'type': 'float', 'description': 'Elitism Rate',
                                 'help': 'Proportion of elite individuals (0.0-1.0)'}
            }

        elif self.algorithm_name == "GreedyAlgorithm":
            definitions = {
                'course_selection_strategy': {'type': 'choice', 'description': 'Course Selection',
                                              'choices': ['profit_density', 'profit_first', 'duration_first',
                                                          'participants_first', 'first_available'],
                                              'help': 'How to select courses'},
                'room_selection_strategy': {'type': 'choice', 'description': 'Room Selection',
                                            'choices': ['cheapest', 'most_expensive', 'smallest', 'largest',
                                                        'first_available'],
                                            'help': 'How to select rooms'},
                'instructor_selection_strategy': {'type': 'choice', 'description': 'Instructor Selection',
                                                  'choices': ['cheapest', 'most_expensive', 'first_available'],
                                                  'help': 'How to select instructors'},
                'participant_strategy': {'type': 'choice', 'description': 'Participant Strategy',
                                         'choices': ['maximum', 'minimum', 'random', 'adaptive'],
                                         'help': 'How to set number of participants'}
            }

        elif self.algorithm_name == "SimulatedAnnealingAlgorithm":
            definitions = {
                'initial_temperature': {'type': 'float', 'description': 'Initial Temperature',
                                        'help': 'Starting temperature'},
                'final_temperature': {'type': 'float', 'description': 'Final Temperature',
                                      'help': 'Ending temperature'},
                'cooling_rate': {'type': 'float', 'description': 'Cooling Rate',
                                 'help': 'Temperature reduction rate (0.0-1.0)'},
                'cooling_schedule': {'type': 'choice', 'description': 'Cooling Schedule',
                                     'choices': ['geometric', 'linear', 'logarithmic', 'adaptive'],
                                     'help': 'How temperature decreases'},
                'max_iterations': {'type': 'integer', 'description': 'Max Iterations',
                                   'help': 'Maximum iterations per temperature'},
                'reheat_threshold': {'type': 'integer', 'description': 'Reheat Threshold',
                                     'help': 'Iterations before reheating'},
                'reheat_factor': {'type': 'float', 'description': 'Reheat Factor',
                                  'help': 'Temperature increase factor for reheating'}
            }

        # Add common parameters
        common_definitions = {
            'max_iterations': {'type': 'integer', 'description': 'Max Iterations',
                               'help': 'Maximum algorithm iterations'},
            'max_time': {'type': 'float', 'description': 'Max Time (seconds)', 'help': 'Maximum execution time'},
            'target_objective': {'type': 'float', 'description': 'Target Objective',
                                 'help': 'Stop when this objective is reached'},
            'verbose': {'type': 'boolean', 'description': 'Verbose Output',
                        'help': 'Print detailed progress information'}
        }

        definitions.update(common_definitions)
        return definitions

    def apply_clicked(self):
        """Apply parameter changes."""
        try:
            new_params = {}

            for param_name, param_info in self.param_vars.items():
                var = param_info['var']
                param_type = param_info['type']

                value = var.get()

                # Convert value to correct type
                if param_type == 'boolean':
                    new_params[param_name] = value
                elif param_type == 'integer':
                    new_params[param_name] = int(value)
                elif param_type == 'float':
                    new_params[param_name] = float(value)
                else:
                    new_params[param_name] = value

            # Keep original name if it exists
            if 'name' in self.current_params:
                new_params['name'] = self.current_params['name']

            self.result = new_params
            self.dialog.destroy()

        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please check parameter values:\n{e}")

    def reset_clicked(self):
        """Reset parameters to defaults."""
        # This would require getting default parameters again
        messagebox.showinfo("Reset", "Reset to defaults not implemented yet")

    def cancel_clicked(self):
        """Cancel parameter changes."""
        self.dialog.destroy()





class ClientDialog:
    """Dialog for adding/editing clients."""

    def __init__(self, parent, client_data=None, title="Client"):
        self.result = None
        self.client_data = client_data

        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.resizable(False, False)

        # Center dialog
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 100, parent.winfo_rooty() + 100))

        self.create_widgets()

    def create_widgets(self):
        """Create dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Client Information",
                                font=('TkDefaultFont', 12, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Form fields
        ttk.Label(main_frame, text="First Name:").grid(row=1, column=0, sticky=tk.W, pady=8)
        self.first_name_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.first_name_var, width=30).grid(row=1, column=1, pady=8, padx=(10, 0))

        ttk.Label(main_frame, text="Last Name:").grid(row=2, column=0, sticky=tk.W, pady=8)
        self.last_name_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.last_name_var, width=30).grid(row=2, column=1, pady=8, padx=(10, 0))

        ttk.Label(main_frame, text="Email:").grid(row=3, column=0, sticky=tk.W, pady=8)
        self.email_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.email_var, width=30).grid(row=3, column=1, pady=8, padx=(10, 0))

        ttk.Label(main_frame, text="Phone:").grid(row=4, column=0, sticky=tk.W, pady=8)
        self.phone_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.phone_var, width=30).grid(row=4, column=1, pady=8, padx=(10, 0))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)

        ttk.Button(button_frame, text="Save", command=self.save_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side=tk.LEFT, padx=5)

        # Load existing data if editing
        if self.client_data:
            self.first_name_var.set(self.client_data.get('first_name', ''))
            self.last_name_var.set(self.client_data.get('last_name', ''))
            self.email_var.set(self.client_data.get('email', ''))
            self.phone_var.set(self.client_data.get('phone', ''))

        # Focus on first field
        main_frame.children['!entry'].focus_set()

    def save_clicked(self):
        """Handle Save button click."""
        first_name = self.first_name_var.get().strip()
        last_name = self.last_name_var.get().strip()
        email = self.email_var.get().strip()
        phone = self.phone_var.get().strip()

        # Validation
        if not first_name or not last_name:
            messagebox.showwarning("Validation Error", "First name and last name are required")
            return

        self.result = {
            'first_name': first_name,
            'last_name': last_name,
            'email': email if email else None,
            'phone': phone if phone else None
        }

        self.dialog.destroy()

    def cancel_clicked(self):
        """Handle Cancel button click."""
        self.dialog.destroy()


class ReservationDialog:
    """Dialog for adding/editing reservations."""

    def __init__(self, parent, clients, courses, reservation_data=None, title="Reservation"):
        self.result = None
        self.clients = clients
        self.courses = courses
        self.reservation_data = reservation_data

        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("450x350")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.resizable(False, False)

        # Center dialog
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 100, parent.winfo_rooty() + 100))

        self.create_widgets()

    def create_widgets(self):
        """Create dialog widgets."""
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Reservation Details",
                                font=('TkDefaultFont', 12, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Client selection
        ttk.Label(main_frame, text="Client:").grid(row=1, column=0, sticky=tk.W, pady=8)
        self.client_var = tk.StringVar()
        client_values = [f"{c['id']}: {c['first_name']} {c['last_name']}" for c in self.clients]
        client_combo = ttk.Combobox(main_frame, textvariable=self.client_var,
                                    values=client_values, state="readonly", width=35)
        client_combo.grid(row=1, column=1, pady=8, padx=(10, 0))

        # Course selection
        ttk.Label(main_frame, text="Course:").grid(row=2, column=0, sticky=tk.W, pady=8)
        self.course_var = tk.StringVar()
        course_values = [f"{c['course_id']}: {c['name']}" for c in self.courses]
        course_combo = ttk.Combobox(main_frame, textvariable=self.course_var,
                                    values=course_values, state="readonly", width=35)
        course_combo.grid(row=2, column=1, pady=8, padx=(10, 0))

        # Status selection
        ttk.Label(main_frame, text="Status:").grid(row=3, column=0, sticky=tk.W, pady=8)
        self.status_var = tk.StringVar(value="pending")
        status_combo = ttk.Combobox(main_frame, textvariable=self.status_var,
                                    values=["pending", "confirmed", "cancelled"],
                                    state="readonly", width=35)
        status_combo.grid(row=3, column=1, pady=8, padx=(10, 0))

        # Notes
        ttk.Label(main_frame, text="Notes:").grid(row=4, column=0, sticky=tk.W+tk.N, pady=8)
        self.notes_text = tk.Text(main_frame, height=4, width=35)
        self.notes_text.grid(row=4, column=1, pady=8, padx=(10, 0))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)

        ttk.Button(button_frame, text="Save", command=self.save_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side=tk.LEFT, padx=5)

        # Load existing data if editing
        if self.reservation_data:
            # Set selections based on existing data
            pass  # Implementation for editing

    def save_clicked(self):
        """Handle Save button click."""
        client_selection = self.client_var.get()
        course_selection = self.course_var.get()
        status = self.status_var.get()
        notes = self.notes_text.get("1.0", tk.END).strip()

        # Validation
        if not client_selection or not course_selection:
            messagebox.showwarning("Validation Error", "Please select client and course")
            return

        # Extract IDs from selections
        client_id = int(client_selection.split(':')[0])
        course_id = int(course_selection.split(':')[0])

        self.result = {
            'client_id': client_id,
            'course_id': course_id,
            'status': status,
            'notes': notes if notes else None
        }

        self.dialog.destroy()

    def cancel_clicked(self):
        """Handle Cancel button click."""
        self.dialog.destroy()


# MAIN FUNCTION
# ============================================================================

def main():
    """Run the GUI application."""
    try:
        # Set up logging
        logger.info("Starting Course Scheduling GUI")

        # Create and run application
        app = CourseSchedulingGUI()
        app.run()

    except Exception as e:
        logger.error(f"Critical error starting GUI: {e}", exc_info=True)
        print(f"Error starting GUI: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()