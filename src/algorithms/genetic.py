import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

from models.problem import CourseSchedulingProblem
from models.solution import Solution
from base import ImprovementAlgorithm


class SelectionMethod(Enum):
    """Selection methods for choosing parents."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    ELITIST = "elitist"


class CrossoverMethod(Enum):
    """Crossover methods for creating offspring."""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    COURSE_BASED = "course_based"


class MutationMethod(Enum):
    """Mutation methods for introducing variation."""
    SWAP_COURSES = "swap_courses"
    REASSIGN_COURSE = "reassign_course"
    RANDOM_CHANGE = "random_change"
    ADAPTIVE = "adaptive"


@dataclass
class Individual:
    """Represents an individual in the genetic algorithm population."""
    solution: Solution
    fitness: float = 0.0
    age: int = 0
    is_feasible: bool = False

    def copy(self) -> 'Individual':
        """Create a deep copy of the individual."""
        return Individual(
            solution=self.solution.copy(),
            fitness=self.fitness,
            age=self.age,
            is_feasible=self.is_feasible
        )


class GeneticAlgorithm(ImprovementAlgorithm):
    """
    Genetic Algorithm for course scheduling optimization.

    Uses evolutionary principles to evolve a population of solutions
    toward better fitness values through selection, crossover, and mutation.
    """

    def __init__(self, problem: CourseSchedulingProblem, **kwargs):
        """
        Initialize genetic algorithm.

        Args:
            problem: Problem instance
            **kwargs: Algorithm parameters
        """
        # Extract name from kwargs or use default
        algorithm_name = kwargs.pop('name', 'GeneticAlgorithm')
        super().__init__(problem, name=algorithm_name, **kwargs)

        # Population parameters
        self.population_size = kwargs.get('population_size', 50)
        self.generations = kwargs.get('generations', 100)

        # Genetic operators parameters
        self.crossover_rate = kwargs.get('crossover_rate', 0.8)
        self.mutation_rate = kwargs.get('mutation_rate', 0.1)
        self.elitism_count = kwargs.get('elitism_count', 2)

        # Selection parameters
        self.selection_method = SelectionMethod(kwargs.get('selection_method', 'tournament'))
        self.tournament_size = kwargs.get('tournament_size', 3)

        # Crossover parameters
        self.crossover_method = CrossoverMethod(kwargs.get('crossover_method', 'course_based'))

        # Mutation parameters
        self.mutation_method = MutationMethod(kwargs.get('mutation_method', 'adaptive'))
        self.mutation_strength = kwargs.get('mutation_strength', 0.5)

        # Diversity and convergence
        self.diversity_preservation = kwargs.get('diversity_preservation', True)
        self.convergence_threshold = kwargs.get('convergence_threshold', 0.01)
        self.stagnation_limit = kwargs.get('stagnation_limit', 20)

        # Random seed
        self.random_seed = kwargs.get('random_seed', None)
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        # Population tracking
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.generation = 0
        self.stagnation_counter = 0

        # Statistics
        self.fitness_history = []
        self.diversity_history = []

    def solve(self) -> Solution:
        """
        Main solving method implementing genetic algorithm.

        Returns:
            Best solution found
        """
        if self.verbose:
            print(f"Starting genetic algorithm:")
            print(f"  Population size: {self.population_size}")
            print(f"  Generations: {self.generations}")
            print(f"  Crossover rate: {self.crossover_rate}")
            print(f"  Mutation rate: {self.mutation_rate}")
            print(f"  Selection: {self.selection_method.value}")
            print(f"  Crossover: {self.crossover_method.value}")
            print(f"  Mutation: {self.mutation_method.value}")

        # Initialize population
        self._initialize_population()

        # Evolution loop
        for generation in range(self.generations):
            self.generation = generation

            # Evaluate population
            self._evaluate_population()

            # Track statistics
            self._update_statistics()

            # Check termination criteria
            if self._should_terminate():
                if self.verbose:
                    print(f"  Early termination at generation {generation}")
                break

            # Create next generation
            new_population = self._create_next_generation()

            # Replace population
            self.population = new_population

            # Log progress
            if self.best_individual:
                self.log_progress(
                    generation + 1,
                    self.best_individual.fitness,
                    f"(feasible: {self.best_individual.is_feasible}, "
                    f"diversity: {self._calculate_diversity():.3f})"
                )

        # ... existing code ...

        if self.verbose:
            print(f"\nGenetic algorithm completed:")
            print(f"  Final generation: {self.generation}")
            if self.best_individual:
                print(f"  Best fitness: {self.best_individual.fitness:.2f}")
                print(f"  Best feasible: {self.best_individual.is_feasible}")
                self.best_individual.solution.print_solution_summary()

        # DODAJ TO - ustaw best_solution w klasie bazowej
        if self.best_individual:
            self.best_solution = self.best_individual.solution
            # Oblicz rzeczywistą wartość objective (bez kar)
            x, y, u = self.best_solution.to_arrays()
            self.stats.best_objective = self.problem.calculate_objective(x, y, u)
            self.stats.final_objective = self.stats.best_objective

        return self.best_individual.solution if self.best_individual else self.create_empty_solution()

    def generate_initial_solution(self) -> Solution:
        """Ulepszona inicjalizacja - najpierw wielogodzinne."""
        solution = self.create_empty_solution()

        # Sortuj kursy: najpierw wielogodzinne, potem według profit density
        course_priorities = []
        for course_id in range(1, self.problem.N + 1):
            course_idx = course_id - 1
            duration = self.problem.course_durations[course_idx]
            density = self.problem.get_course_profit_density(course_id)

            # Kombinuj duration i profit density - priorytet dla wielogodzinnych
            priority_score = duration * 1000 + density  # Wielogodzinne mają przewagę
            course_priorities.append((course_id, priority_score))

        course_priorities.sort(key=lambda x: x[1], reverse=True)
        course_order = [course_id for course_id, _ in course_priorities]

        for course_id in course_order:
            course_idx = course_id - 1
            duration = self.problem.course_durations[course_idx]

            participants = self.problem.get_actual_participants(course_id)
            if participants == 0:
                participants = self.problem.course_max_participants[course_idx]

            # Znajdź kompatybilne zasoby
            compatible_rooms = []
            for room_id in range(1, self.problem.M + 1):
                room_idx = room_id - 1
                if (self.problem.R_matrix[course_idx, room_idx] == 1 and
                        self.problem.room_capacities[room_idx] >= participants):
                    compatible_rooms.append(room_id)

            compatible_instructors = []
            for instructor_id in range(1, self.problem.K + 1):
                instructor_idx = instructor_id - 1
                if self.problem.I_matrix[instructor_idx, course_idx] == 1:
                    compatible_instructors.append(instructor_id)

            if not compatible_rooms or not compatible_instructors:
                continue

            # Dla kursów wielogodzinnych - szukaj ciągłych bloków
            success = False
            if duration > 1:
                # Systematyczne przeszukiwanie ciągłych bloków
                for start_time in range(1, self.problem.T - duration + 2):
                    for room_id in compatible_rooms:
                        for instructor_id in compatible_instructors:
                            # Sprawdź dostępność całego bloku
                            can_assign = True
                            instructor_idx = instructor_id - 1

                            for t_offset in range(duration):
                                t = start_time + t_offset
                                if (not solution.is_room_available(room_id, t) or
                                        not solution.is_instructor_available(instructor_id, t) or
                                        t - 1 >= self.problem.T or
                                        self.problem.A_matrix[instructor_idx, t - 1] == 0):
                                    can_assign = False
                                    break

                            if can_assign:
                                if solution.assign_course(course_id, room_id, instructor_id,
                                                          start_time, duration, participants):
                                    success = True
                                    break
                        if success:
                            break
                    if success:
                        break
            else:
                # Dla kursów jednogodzinnych - standardowo
                for attempt in range(50):
                    room_id = random.choice(compatible_rooms)
                    instructor_id = random.choice(compatible_instructors)
                    start_time = random.randint(1, self.problem.T)

                    instructor_idx = instructor_id - 1
                    if (solution.is_room_available(room_id, start_time) and
                            solution.is_instructor_available(instructor_id, start_time) and
                            start_time - 1 < self.problem.T and
                            self.problem.A_matrix[instructor_idx, start_time - 1] == 1):

                        if solution.assign_course(course_id, room_id, instructor_id,
                                                  start_time, duration, participants):
                            break

        return solution

    def improve_solution(self, solution: Solution) -> Solution:
        """
        Improve solution (not used in standard GA, but required by base class).

        Args:
            solution: Solution to improve

        Returns:
            Improved solution
        """
        # In GA, improvement happens through evolution, not direct improvement
        return solution

    def _initialize_population(self) -> None:
        """Initialize the population with random individuals."""
        if self.verbose:
            print("  Initializing population...")

        self.population = []

        for i in range(self.population_size):
            solution = self.generate_initial_solution()
            individual = Individual(solution=solution)
            self.population.append(individual)

        # Evaluate initial population
        self._evaluate_population()

        if self.verbose:
            feasible_count = sum(1 for ind in self.population if ind.is_feasible)
            print(f"  Initial population: {feasible_count}/{self.population_size} feasible")

    def _evaluate_population(self) -> None:
        """Evaluate fitness for all individuals in population."""
        for individual in self.population:
            x, y, u = individual.solution.to_arrays()

            # Check feasibility
            individual.is_feasible = self.problem.is_feasible(x, y, u)

            # Calculate fitness
            objective = self.problem.calculate_objective(x, y, u)

            # Penalty for infeasible solutions
            if not individual.is_feasible:
                # Count violations for penalty
                _, violations = self.problem.validate_solution(x, y, u)
                penalty = len(violations) * 100  # light penalty
                individual.fitness = objective - penalty
            else:
                individual.fitness = objective

            # Update statistics
            self.stats.evaluations += 1

            # Track best individual - ZMIANA TUTAJ!
            # Zamiast warunku na feasibility, zapisuj najlepsze rozwiązanie niezależnie od feasibility
            if (self.best_individual is None or
                    individual.fitness > self.best_individual.fitness):
                self.best_individual = individual
                self.stats.improvements += 1

                # Dodatkowo śledź feasible solutions
                if individual.is_feasible:
                    self.stats.feasible_solutions_found += 1

    def _create_next_generation(self) -> List[Individual]:
        """Ulepszona wersja z walidacją."""
        new_population = []

        # Elitizm
        if self.elitism_count > 0:
            elite = self._select_elite(self.elitism_count)
            new_population.extend([ind.copy() for ind in elite])

        # Generuj potomstwo
        while len(new_population) < self.population_size:
            parent1 = self._select_parent()
            parent2 = self._select_parent()

            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutacja z walidacją
            if random.random() < self.mutation_rate:
                child1 = self._mutate(child1)
                # Sprawdź czy mutacja nie zepsuła rozwiązania
                if len(child1.solution.get_scheduled_courses()) < len(parent1.solution.get_scheduled_courses()):
                    child1 = parent1.copy()  # Przywróć rodzica jeśli mutacja pogorszyła

            if random.random() < self.mutation_rate:
                child2 = self._mutate(child2)
                if len(child2.solution.get_scheduled_courses()) < len(parent2.solution.get_scheduled_courses()):
                    child2 = parent2.copy()

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        return new_population[:self.population_size]

    def _assign_single_hour_course(self, solution: Solution, assignment: Dict) -> bool:
        """Metoda dla kursów jednogodzinnych."""
        course_id = assignment['course_id']
        duration = assignment['duration']
        participants = assignment['participants']
        course_idx = course_id - 1

        # Znajdź alternatywne sale i instruktorów
        compatible_rooms = []
        for room_id in range(1, self.problem.M + 1):
            room_idx = room_id - 1
            if (self.problem.R_matrix[course_idx, room_idx] == 1 and
                    self.problem.room_capacities[room_idx] >= participants):
                compatible_rooms.append(room_id)

        compatible_instructors = []
        for instructor_id in range(1, self.problem.K + 1):
            instructor_idx = instructor_id - 1
            if self.problem.I_matrix[instructor_idx, course_idx] == 1:
                compatible_instructors.append(instructor_id)

        # Spróbuj wszystkie kombinacje
        for room_id in compatible_rooms:
            for instructor_id in compatible_instructors:
                for start_time in range(1, self.problem.T + 1):
                    # Sprawdź dostępność
                    if (solution.is_room_available(room_id, start_time) and
                            solution.is_instructor_available(instructor_id, start_time)):

                        # Sprawdź również dostępność instruktora w A_matrix
                        instructor_idx = instructor_id - 1
                        t_idx = start_time - 1
                        if t_idx < self.problem.T and self.problem.A_matrix[instructor_idx, t_idx] == 1:
                            if solution.assign_course(
                                    course_id, room_id, instructor_id,
                                    start_time, duration, participants
                            ):
                                return True

        return False

    def _select_parent(self) -> Individual:
        """Select parent using configured selection method."""
        if self.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection()
        elif self.selection_method == SelectionMethod.ROULETTE:
            return self._roulette_selection()
        elif self.selection_method == SelectionMethod.RANK:
            return self._rank_selection()
        else:  # ELITIST
            return self._elitist_selection()

    def _tournament_selection(self) -> Individual:
        """Tournament selection."""
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(tournament, key=lambda ind: ind.fitness)

    def _roulette_selection(self) -> Individual:
        """Roulette wheel selection."""
        # Shift fitness to positive values
        min_fitness = min(ind.fitness for ind in self.population)
        adjusted_fitness = [ind.fitness - min_fitness + 1 for ind in self.population]

        total_fitness = sum(adjusted_fitness)
        if total_fitness == 0:
            return random.choice(self.population)

        pick = random.uniform(0, total_fitness)
        current = 0
        for i, fitness in enumerate(adjusted_fitness):
            current += fitness
            if current >= pick:
                return self.population[i]

        return self.population[-1]  # Fallback

    def _rank_selection(self) -> Individual:
        """Rank-based selection."""
        sorted_pop = sorted(self.population, key=lambda ind: ind.fitness)
        ranks = list(range(1, len(sorted_pop) + 1))
        total_rank = sum(ranks)

        pick = random.uniform(0, total_rank)
        current = 0
        for i, rank in enumerate(ranks):
            current += rank
            if current >= pick:
                return sorted_pop[i]

        return sorted_pop[-1]  # Fallback

    def _elitist_selection(self) -> Individual:
        """Elitist selection - always pick from top performers."""
        top_count = max(1, len(self.population) // 4)
        top_individuals = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)[:top_count]
        return random.choice(top_individuals)

    def _select_elite(self, count: int) -> List[Individual]:
        """Select elite individuals."""
        sorted_pop = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
        return sorted_pop[:count]

    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        if self.crossover_method == CrossoverMethod.SINGLE_POINT:
            return self._single_point_crossover(parent1, parent2)
        elif self.crossover_method == CrossoverMethod.TWO_POINT:
            return self._two_point_crossover(parent1, parent2)
        elif self.crossover_method == CrossoverMethod.UNIFORM:
            return self._uniform_crossover(parent1, parent2)
        else:  # COURSE_BASED
            return self._course_based_crossover(parent1, parent2)

    def _course_based_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Ulepszona wersja uwzględniająca kursy wielogodzinne."""
        child1 = Individual(solution=self.create_empty_solution())
        child2 = Individual(solution=self.create_empty_solution())

        # Pogrupuj kursy według długości - najpierw wielogodzinne
        assignments1 = parent1.solution.get_all_assignments()
        assignments2 = parent2.solution.get_all_assignments()

        # Sortuj według duration DESC - najpierw długie kursy
        assignments1.sort(key=lambda x: x['duration'], reverse=True)
        assignments2.sort(key=lambda x: x['duration'], reverse=True)

        # Przypisuj kursy zachowując priorytet dla wielogodzinnych
        all_courses = set(a['course_id'] for a in assignments1 + assignments2)

        for course_id in sorted(all_courses, key=lambda cid: self.problem.course_durations[cid - 1], reverse=True):
            # Większe prawdopodobieństwo dla lepszego rodzica
            parent1_better = parent1.fitness > parent2.fitness
            take_from_parent1 = random.random() < (0.7 if parent1_better else 0.3)

            assignment1 = next((a for a in assignments1 if a['course_id'] == course_id), None)
            assignment2 = next((a for a in assignments2 if a['course_id'] == course_id), None)

            if take_from_parent1 and assignment1:
                self._try_assign_with_alternatives(child1.solution, assignment1)
                if assignment2:
                    self._try_assign_with_alternatives(child2.solution, assignment2)
            elif assignment2:
                self._try_assign_with_alternatives(child1.solution, assignment2)
                if assignment1:
                    self._try_assign_with_alternatives(child2.solution, assignment1)

        return child1, child2

    def _single_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single-point crossover z obsługą kursów wielogodzinnych."""
        assignments1 = parent1.solution.get_all_assignments()
        assignments2 = parent2.solution.get_all_assignments()

        # DODANE: Sortuj według duration DESC
        assignments1.sort(key=lambda x: x['duration'], reverse=True)
        assignments2.sort(key=lambda x: x['duration'], reverse=True)

        max_len = max(len(assignments1), len(assignments2))
        if max_len == 0:
            return parent1.copy(), parent2.copy()

        crossover_point = random.randint(1, max_len)

        child1 = Individual(solution=self.create_empty_solution())
        child2 = Individual(solution=self.create_empty_solution())

        # Apply crossover Z POPRAWKAMI
        for i, assignment in enumerate(assignments1):
            if i < crossover_point:
                self._try_assign_with_alternatives(child1.solution, assignment)  # ZMIANA
            else:
                self._try_assign_with_alternatives(child2.solution, assignment)  # ZMIANA

        for i, assignment in enumerate(assignments2):
            if i < crossover_point:
                self._try_assign_with_alternatives(child2.solution, assignment)  # ZMIANA
            else:
                self._try_assign_with_alternatives(child1.solution, assignment)  # ZMIANA

        return child1, child2

    def _two_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Two-point crossover z obsługą kursów wielogodzinnych."""
        assignments1 = parent1.solution.get_all_assignments()
        assignments2 = parent2.solution.get_all_assignments()

        # DODANE: Sortuj według duration DESC
        assignments1.sort(key=lambda x: x['duration'], reverse=True)
        assignments2.sort(key=lambda x: x['duration'], reverse=True)

        max_len = max(len(assignments1), len(assignments2))
        if max_len <= 2:
            return self._course_based_crossover(parent1, parent2)  # Fallback

        point1 = random.randint(1, max_len - 1)
        point2 = random.randint(point1 + 1, max_len)

        child1 = Individual(solution=self.create_empty_solution())
        child2 = Individual(solution=self.create_empty_solution())

        # Apply two-point crossover logic Z POPRAWKAMI
        for i, assignment in enumerate(assignments1):
            if i < point1 or i >= point2:
                self._try_assign_with_alternatives(child1.solution, assignment)  # ZMIANA
            else:
                self._try_assign_with_alternatives(child2.solution, assignment)  # ZMIANA

        for i, assignment in enumerate(assignments2):
            if i < point1 or i >= point2:
                self._try_assign_with_alternatives(child2.solution, assignment)  # ZMIANA
            else:
                self._try_assign_with_alternatives(child1.solution, assignment)  # ZMIANA

        return child1, child2

    def _safe_apply_assignment(self, solution: Solution, assignment: Dict) -> bool:
        """Safely apply assignment, handling conflicts by trying alternatives."""
        # Najpierw spróbuj oryginalne przypisanie
        success = solution.assign_course(
            assignment['course_id'],
            assignment['room_id'],
            assignment['instructor_id'],
            assignment['start_time'],
            assignment['duration'],
            assignment['participants']
        )

        if success:
            return True

        # Jeśli nie udało się, spróbuj alternatywne czasy dla tej samej sali/instruktora
        course_id = assignment['course_id']
        course_idx = course_id - 1
        duration = self.problem.course_durations[course_idx]

        # Spróbuj 5 innych czasów
        for attempt in range(5):
            alt_start_time = random.randint(1, max(1, self.problem.T - duration + 1))

            success = solution.assign_course(
                course_id,
                assignment['room_id'],
                assignment['instructor_id'],
                alt_start_time,
                duration,
                assignment['participants']
            )

            if success:
                return True

        # Jeśli nadal się nie udało, spróbuj inną salę/instruktora
        for attempt in range(3):
            # Znajdź alternatywne zasoby
            compatible_rooms = []
            for room_id in range(1, self.problem.M + 1):
                room_idx = room_id - 1
                if (self.problem.R_matrix[course_idx, room_idx] == 1 and
                        self.problem.room_capacities[room_idx] >= assignment['participants']):
                    compatible_rooms.append(room_id)

            compatible_instructors = []
            for instructor_id in range(1, self.problem.K + 1):
                instructor_idx = instructor_id - 1
                if self.problem.I_matrix[instructor_idx, course_idx] == 1:
                    compatible_instructors.append(instructor_id)

            if compatible_rooms and compatible_instructors:
                alt_room = random.choice(compatible_rooms)
                alt_instructor = random.choice(compatible_instructors)
                alt_start_time = random.randint(1, max(1, self.problem.T - duration + 1))

                success = solution.assign_course(
                    course_id, alt_room, alt_instructor, alt_start_time,
                    duration, assignment['participants']
                )

                if success:
                    return True

        return False  # Nie udało się w ogóle

    def _uniform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Improved uniform crossover z poprawką duration."""
        child1 = Individual(solution=self.create_empty_solution())
        child2 = Individual(solution=self.create_empty_solution())

        assignments1 = parent1.solution.get_all_assignments()
        assignments2 = parent2.solution.get_all_assignments()

        # POPRAWKA: Sortuj według duration DESC - najpierw wielogodzinne
        assignments1.sort(key=lambda x: x['duration'], reverse=True)
        assignments2.sort(key=lambda x: x['duration'], reverse=True)

        # Create course assignment maps
        assign_map1 = {a['course_id']: a for a in assignments1}
        assign_map2 = {a['course_id']: a for a in assignments2}

        all_courses = set(assign_map1.keys()) | set(assign_map2.keys())

        # POPRAWKA: Przetwarzaj w kolejności duration DESC
        sorted_courses = sorted(all_courses,
                                key=lambda cid: self.problem.course_durations[cid - 1],
                                reverse=True)

        parent1_prob = 0.5
        if parent1.fitness != parent2.fitness:
            total_fitness = abs(parent1.fitness) + abs(parent2.fitness)
            if total_fitness > 0:
                parent1_prob = abs(parent1.fitness) / total_fitness
                parent1_prob = max(0.2, min(0.8, parent1_prob))

        for course_id in sorted_courses:  # ZMIANA: używaj sorted_courses
            if random.random() < parent1_prob:
                if course_id in assign_map1:
                    self._try_assign_with_alternatives(child1.solution, assign_map1[course_id])
                if course_id in assign_map2:
                    self._try_assign_with_alternatives(child2.solution, assign_map2[course_id])
            else:
                if course_id in assign_map2:
                    self._try_assign_with_alternatives(child1.solution, assign_map2[course_id])
                if course_id in assign_map1:
                    self._try_assign_with_alternatives(child2.solution, assign_map1[course_id])

        return child1, child2

    def _apply_assignment(self, solution: Solution, assignment: Dict) -> bool:
        """Apply assignment to solution, handling conflicts."""
        return solution.assign_course(
            assignment['course_id'],
            assignment['room_id'],
            assignment['instructor_id'],
            assignment['start_time'],
            assignment['duration'],
            assignment['participants']
        )

    def _mutate(self, individual: Individual) -> Individual:
        """Perform mutation on individual."""
        if self.mutation_method == MutationMethod.SWAP_COURSES:
            return self._swap_mutation(individual)
        elif self.mutation_method == MutationMethod.REASSIGN_COURSE:
            return self._reassign_mutation(individual)
        elif self.mutation_method == MutationMethod.RANDOM_CHANGE:
            return self._random_mutation(individual)
        else:  # ADAPTIVE
            return self._adaptive_mutation(individual)

    def _swap_mutation(self, individual: Individual) -> Individual:
        """Ulepszona swap mutation z walidacją duration."""
        mutated = individual.copy()
        assignments = mutated.solution.get_all_assignments()

        if len(assignments) >= 2:
            # ZMIANA: Wybieraj tylko kursy o tym samym duration dla swap
            single_hour = [a for a in assignments if a['duration'] == 1]
            multi_hour = [a for a in assignments if a['duration'] > 1]

            # Swap w obrębie tej samej kategorii duration
            if len(single_hour) >= 2:
                assign1, assign2 = random.sample(single_hour, 2)
                mutated.solution.swap_courses(assign1['course_id'], assign2['course_id'])
            elif len(multi_hour) >= 2:
                assign1, assign2 = random.sample(multi_hour, 2)
                mutated.solution.swap_courses(assign1['course_id'], assign2['course_id'])

        return mutated

    def _reassign_mutation(self, individual: Individual) -> Individual:
        """Reassign a random course to different resources."""
        mutated = individual.copy()
        assignments = mutated.solution.get_all_assignments()

        if assignments:
            # Pick random assignment to modify
            assignment = random.choice(assignments)
            course_id = assignment['course_id']

            # Remove current assignment
            mutated.solution.remove_course_assignment(course_id)

            # Try to reassign with different parameters
            course_idx = course_id - 1
            duration = self.problem.course_durations[course_idx]
            participants = assignment['participants']

            # Find new resources
            for attempt in range(10):
                # Random room and instructor
                room_candidates = []
                instructor_candidates = []

                for room_id in range(1, self.problem.M + 1):
                    room_idx = room_id - 1
                    if (self.problem.R_matrix[course_idx, room_idx] == 1 and
                            self.problem.room_capacities[room_idx] >= participants):
                        room_candidates.append(room_id)

                for instructor_id in range(1, self.problem.K + 1):
                    instructor_idx = instructor_id - 1
                    if self.problem.I_matrix[instructor_idx, course_idx] == 1:
                        instructor_candidates.append(instructor_id)

                if room_candidates and instructor_candidates:
                    room_id = random.choice(room_candidates)
                    instructor_id = random.choice(instructor_candidates)
                    start_time = random.randint(1, max(1, self.problem.T - duration))

                    if mutated.solution.assign_course(course_id, room_id, instructor_id,
                                                      start_time, duration, participants):
                        break

        return mutated

    def _random_mutation(self, individual: Individual) -> Individual:
        """Random mutation - various random changes."""
        mutated = individual.copy()

        # Choose random mutation type
        mutation_type = random.choice(['swap', 'reassign', 'remove_add'])

        if mutation_type == 'swap':
            return self._swap_mutation(individual)
        elif mutation_type == 'reassign':
            return self._reassign_mutation(individual)
        else:  # remove_add
            assignments = mutated.solution.get_all_assignments()
            if assignments:
                # Remove random course
                assignment = random.choice(assignments)
                mutated.solution.remove_course_assignment(assignment['course_id'])

                # Try to add a random unscheduled course
                unscheduled = mutated.solution.get_unscheduled_courses()
                if unscheduled:
                    course_id = random.choice(list(unscheduled))
                    # Try basic assignment (simplified)
                    course_idx = course_id - 1
                    duration = self.problem.course_durations[course_idx]
                    participants = self.problem.course_max_participants[course_idx]

                    for attempt in range(5):
                        room_id = random.randint(1, self.problem.M)
                        instructor_id = random.randint(1, self.problem.K)
                        start_time = random.randint(1, max(1, self.problem.T - duration))

                        if mutated.solution.assign_course(course_id, room_id, instructor_id,
                                                          start_time, duration, participants):
                            break

        return mutated

    def _try_assign_with_alternatives(self, solution: Solution, assignment: Dict) -> bool:
        """Spróbuj przypisać kurs z inteligentnymi alternatywami."""
        course_id = assignment['course_id']
        duration = assignment['duration']

        # Najpierw spróbuj oryginalne przypisanie
        if solution.assign_course(
                course_id, assignment['room_id'], assignment['instructor_id'],
                assignment['start_time'], duration, assignment['participants']
        ):
            return True

        # Dla kursów wielogodzinnych - znajdź ciągłe bloki czasowe
        if duration > 1:
            return self._assign_multi_hour_course(solution, assignment)
        else:
            return self._assign_single_hour_course(solution, assignment)

    def _assign_multi_hour_course(self, solution: Solution, assignment: Dict) -> bool:
        """Specjalna metoda dla kursów wielogodzinnych."""
        course_id = assignment['course_id']
        duration = assignment['duration']
        participants = assignment['participants']

        # Znajdź wszystkie możliwe ciągłe bloki czasowe
        for start_time in range(1, self.problem.T - duration + 2):
            # Sprawdź czy można przypisać w tym bloku
            can_assign = True

            # Sprawdź dostępność sali i instruktora dla całego bloku
            for t_offset in range(duration):
                t = start_time + t_offset
                if not solution.is_room_available(assignment['room_id'], t):
                    can_assign = False
                    break
                if not solution.is_instructor_available(assignment['instructor_id'], t):
                    can_assign = False
                    break

            if can_assign:
                return solution.assign_course(
                    course_id, assignment['room_id'], assignment['instructor_id'],
                    start_time, duration, participants
                )

        # Jeśli nie udało się z oryginalną salą/instruktorem, spróbuj inne kombinacje
        course_idx = course_id - 1

        # Znajdź alternatywne sale i instruktorów
        compatible_rooms = []
        for room_id in range(1, self.problem.M + 1):
            room_idx = room_id - 1
            if (self.problem.R_matrix[course_idx, room_idx] == 1 and
                    self.problem.room_capacities[room_idx] >= participants):
                compatible_rooms.append(room_id)

        compatible_instructors = []
        for instructor_id in range(1, self.problem.K + 1):
            instructor_idx = instructor_id - 1
            if self.problem.I_matrix[instructor_idx, course_idx] == 1:
                compatible_instructors.append(instructor_id)

        # Spróbuj wszystkie kombinacje dla ciągłych bloków
        for room_id in compatible_rooms:
            for instructor_id in compatible_instructors:
                for start_time in range(1, self.problem.T - duration + 2):
                    can_assign = True

                    # Sprawdź dostępność dla całego bloku
                    for t_offset in range(duration):
                        t = start_time + t_offset
                        if not solution.is_room_available(room_id, t):
                            can_assign = False
                            break
                        if not solution.is_instructor_available(instructor_id, t):
                            can_assign = False
                            break

                        # Sprawdź również dostępność instruktora w A_matrix
                        instructor_idx = instructor_id - 1
                        t_idx = t - 1
                        if t_idx >= self.problem.T or self.problem.A_matrix[instructor_idx, t_idx] == 0:
                            can_assign = False
                            break

                    if can_assign:
                        return solution.assign_course(
                            course_id, room_id, instructor_id,
                            start_time, duration, participants
                        )

        return False

    def _adaptive_mutation(self, individual: Individual) -> Individual:
        """Ulepszona mutacja z mechanizmem naprawy."""
        original_scheduled = len(individual.solution.get_scheduled_courses())
        mutated = individual.copy()

        # Spróbuj najpierw przypisać nieprzypisane kursy
        unscheduled = mutated.solution.get_unscheduled_courses()

        if unscheduled:
            # Sortuj według duration DESC - prioritet dla wielogodzinnych
            unscheduled_sorted = sorted(unscheduled,
                                        key=lambda cid: self.problem.course_durations[cid - 1],
                                        reverse=True)

            for course_id in unscheduled_sorted[:2]:  # Maksymalnie 2 próby
                course_idx = course_id - 1
                duration = self.problem.course_durations[course_idx]
                participants = self.problem.get_actual_participants(course_id)

                dummy_assignment = {
                    'course_id': course_id,
                    'room_id': 1,
                    'instructor_id': 1,
                    'start_time': 1,
                    'duration': duration,
                    'participants': participants if participants > 0 else self.problem.course_max_participants[
                        course_idx]
                }

                if duration > 1:
                    self._assign_multi_hour_course(mutated.solution, dummy_assignment)
                else:
                    self._assign_single_hour_course(mutated.solution, dummy_assignment)

        # Tylko jeśli nie udało się poprawić, rób standardową mutację
        if len(mutated.solution.get_scheduled_courses()) <= original_scheduled:
            if random.random() < 0.2:  # Bardzo ostrożnie
                if random.random() < 0.5:
                    mutated = self._swap_mutation(mutated)
                else:
                    mutated = self._reassign_mutation(mutated)

        # Jeśli mutacja pogorszyła rozwiązanie, przywróć oryginał
        if len(mutated.solution.get_scheduled_courses()) < original_scheduled:
            return individual

        return mutated

    def _update_statistics(self) -> None:
        """Update algorithm statistics."""
        if self.population:
            fitnesses = [ind.fitness for ind in self.population]
            self.fitness_history.append({
                'generation': self.generation,
                'best': max(fitnesses),
                'average': np.mean(fitnesses),
                'worst': min(fitnesses)
            })

            diversity = self._calculate_diversity()
            self.diversity_history.append(diversity)

            # Update custom metrics
            feasible_count = sum(1 for ind in self.population if ind.is_feasible)
            self.stats.custom_metrics.update({
                'diversity': diversity,
                'feasible_ratio': feasible_count / len(self.population),
                'generation': self.generation
            })

    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0

        # Simple diversity measure: average difference in scheduled courses
        total_differences = 0
        comparisons = 0

        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                scheduled1 = self.population[i].solution.get_scheduled_courses()
                scheduled2 = self.population[j].solution.get_scheduled_courses()

                # Calculate Jaccard distance (1 - intersection/union)
                if len(scheduled1) == 0 and len(scheduled2) == 0:
                    difference = 0.0
                else:
                    intersection = len(scheduled1 & scheduled2)
                    union = len(scheduled1 | scheduled2)
                    difference = 1.0 - (intersection / union) if union > 0 else 0.0

                total_differences += difference
                comparisons += 1

        return total_differences / comparisons if comparisons > 0 else 0.0

    def _should_terminate(self) -> bool:
        """Check if algorithm should terminate early."""
        # Check time limit
        if self.should_terminate(self.generation):
            return True

        # Check convergence
        if len(self.fitness_history) >= 2:
            recent_fitness = [entry['best'] for entry in self.fitness_history[-10:]]
            if len(recent_fitness) >= 2:
                improvement = recent_fitness[-1] - recent_fitness[0]
                if abs(improvement) < self.convergence_threshold:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0

                if self.stagnation_counter >= self.stagnation_limit:
                    return True

        return False


def create_genetic_variants() -> List[Dict]:
    """
    Create different genetic algorithm variants with various parameter combinations.

    Returns:
        List of parameter dictionaries for different GA variants
    """
    variants = []

    # Basic GA with tournament selection
    variants.append({
        'name': 'GA_Tournament_CourseBased',
        'population_size': 100,
        'generations': 300,
        'crossover_rate': 0.8,
        'mutation_rate': 0.05,
        'selection_method': 'tournament',
        'tournament_size': 3,
        'crossover_method': 'course_based',
        'mutation_method': 'adaptive',
        'elitism_count': 2
    })

    # GA with roulette selection and uniform crossover
    variants.append({
        'name': 'GA_Roulette_Uniform',
        'population_size': 100,
        'generations': 450,
        'crossover_rate': 0.9,
        'mutation_rate': 0.07,
        'selection_method': 'roulette',
        'crossover_method': 'uniform',
        'mutation_method': 'adaptive',
        'elitism_count': 3
    })

    # High diversity GA
    variants.append({
        'name': 'GA_HighDiversity',
        'population_size': 100,
        'generations': 200,
        'crossover_rate': 0.7,
        'mutation_rate': 0.01,
        'selection_method': 'rank',
        'crossover_method': 'course_based',
        'mutation_method': 'adaptive',
        'elitism_count': 2,
        'diversity_preservation': False
    })

    # Elitist GA with low mutation
    variants.append({
        'name': 'GA_Elitist_LowMutation',
        'population_size': 80,
        'generations': 300,
        'crossover_rate': 0.95,
        'mutation_rate': 0.01,
        'selection_method': 'elitist',
        'crossover_method': 'course_based',
        'mutation_method': 'reassign_course',
        'elitism_count': 5
    })

    # Fast GA for quick results
    variants.append({
        'name': 'GA_Fast',
        'population_size': 100,
        'generations': 100,
        'crossover_rate': 0.8,
        'mutation_rate': 0.1,
        'selection_method': 'tournament',
        'tournament_size': 2,
        'crossover_method': 'course_based',
        'mutation_method': 'adaptive',
        'elitism_count': 1,
        'convergence_threshold': 0.1,
        'stagnation_limit': 10
    })


    variants.append({
        'name': 'GA_Conservative',
        'population_size': 100,
        'generations': 200,
        'crossover_rate': 0.8,
        'mutation_rate': 0.05,
        'selection_method': 'tournament',
        'tournament_size': 3,
        'crossover_method': 'course_based',  # Sprawdzona najlepsza
        'mutation_method': 'adaptive',  # Sprawdzona najlepsza
        'elitism_count': 2
    })

    # 2. HYBRID - Uniform crossover (działał dobrze) + inne parametry
    variants.append({
        'name': 'GA_Hybrid',
        'population_size': 120,
        'generations': 300,
        'crossover_rate': 0.9,
        'mutation_rate': 0.08,
        'selection_method': 'roulette',
        'crossover_method': 'uniform',  # RÓŻNICA: uniform zamiast course_based
        'mutation_method': 'adaptive',  # Sprawdzona
        'elitism_count': 3
    })

    # 3. INTENSIVE - Duże populacje, niskie mutacje, intensywny elityzm
    variants.append({
        'name': 'GA_Intensive',
        'population_size': 150,
        'generations': 250,
        'crossover_rate': 0.95,
        'mutation_rate': 0.02,  # RÓŻNICA: bardzo niska mutacja
        'selection_method': 'elitist',
        'crossover_method': 'course_based',
        'mutation_method': 'reassign_course',  # RÓŻNICA: reassign zamiast adaptive
        'elitism_count': 8  # RÓŻNICA: wysoki elityzm
    })

# 4. EXPLORATORY - Wyższa różnorodność, rank selection
    variants.append({
        'name': 'GA_Exploratory',
        'population_size': 80,
        'generations': 400,
        'crossover_rate': 0.7,  # RÓŻNICA: niższy crossover
        'mutation_rate': 0.12,  # RÓŻNICA: wyższa mutacja
        'selection_method': 'rank',  # RÓŻNICA: rank selection
        'crossover_method': 'two_point',  # RÓŻNICA: powrót do two_point ale z poprawkami
        'mutation_method': 'adaptive',
        'elitism_count': 1,  # RÓŻNICA: minimalny elityzm
        'diversity_preservation': True  # RÓŻNICA: włączona dywersyfikacja
    })

    # 5. FAST_BALANCED - Szybki ale zbalansowany
    variants.append({
        'name': 'GA_FastBalanced',
        'population_size': 60,  # RÓŻNICA: mniejsza populacja
        'generations': 150,  # RÓŻNICA: mniej generacji
        'crossover_rate': 0.85,
        'mutation_rate': 0.15,  # RÓŻNICA: wysoka mutacja dla szybkości
        'selection_method': 'tournament',
        'tournament_size': 2,  # RÓŻNICA: mniejszy turniej
        'crossover_method': 'single_point',  # RÓŻNICA: single_point z poprawkami
        'mutation_method': 'swap_courses',  # RÓŻNICA: powrót do swap ale z poprawkami
        'elitism_count': 1,
        'convergence_threshold': 0.05,
        'stagnation_limit': 15

    })

    return variants


if __name__ == "__main__":
    # Test genetic algorithm
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

        print("Testing Genetic Algorithm variants...")

        # Test different variants
        variants = create_genetic_variants()

        for variant in variants[:2]:  # Test first 2 variants
            print(f"\n{'=' * 60}")
            print(f"Testing: {variant['name']}")
            print(f"{'=' * 60}")

            # Create and run algorithm
            ga = GeneticAlgorithm(problem, verbose=True, **variant)
            solution = ga.run()

            # Print results
            ga.print_statistics()

            # Validate solution
            x, y, u = solution.to_arrays()
            is_feasible, violations = problem.validate_solution(x, y, u)

            print(f"\nSolution validation:")
            print(f"  Feasible: {is_feasible}")
            if violations:
                print(f"  Violations: {len(violations)}")
                for violation in violations[:3]:  # Show first 3
                    print(f"    - {violation}")

            # Print convergence info
            if ga.fitness_history:
                print(f"\nConvergence:")
                print(f"  Initial best: {ga.fitness_history[0]['best']:.2f}")
                print(f"  Final best: {ga.fitness_history[-1]['best']:.2f}")
                print(f"  Final diversity: {ga.diversity_history[-1]:.3f}")

        print(f"\n✓ Genetic algorithm testing completed!")

    except Exception as e:
        print(f"Error testing genetic algorithm: {e}")
        import traceback

        traceback.print_exc()