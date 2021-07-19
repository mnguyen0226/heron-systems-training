"""
Search classes that use a win classifier to evaluate states

Genetic Search:
Generating N states, evaluates each state with the win probability classifier,
selects the good states, duplicates and mutates to get a new set of states to evaluate.
*Future: Store the best solution at the end of each generation; run until a kill signal is send, then return current best solution

Full Search:
Generate states N times, takes all uniques states and evaluates them. Extremely inefficent and slow.
*Future: There are better ways to do this, like finding all possible values for each part of the state and using itertools.product()

Monte Carlo Search:
Generates N random states and evaluates them. Good for finding the appoximate solution quickly and simply.
"""
import random

from deap import base
from deap import creator
from deap import tools

from gamebreaker.search.base import Search
from gamebreaker.search.utils import create_units
from gamebreaker.search.utils import format_ind
from gamebreaker.search.utils import mutate


class GeneticSearch(Search):
    def __init__(
        self,
        network,
        state,
        selector,
        objective=(1.0,),
        npop=50,
        ngen=40,
        cxpb=0.5,
        mutpb=0.2,
        mutnb=1,
    ):
        super().__init__(network, state)
        self.objective = objective
        self.selector = selector
        self.toolbox = self._setup_toolbox()
        self.npop = npop
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.mutnb = mutnb

    def search(self):
        pop = self.toolbox.population(n=self.npop)
        formatted_pop = map(format_ind, pop)
        fitnesses = map(self.toolbox.evaluate, formatted_pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        for _ in range(self.ngen):
            pop[:] = self._run_gen(pop)

        for ind in pop:
            print(ind.fitness.values)
        winner = self.toolbox.winner(pop, 1)[0]
        winner = format_ind(winner)
        return winner, self.classifier.eval(winner)

    def _setup_toolbox(self):
        toolbox = base.Toolbox()
        creator.create("Fitness", base.Fitness, weights=self.objective)
        creator.create("Individual", list, fitness=creator.Fitness)
        toolbox.register("evaluate", self.classifier.eval)
        toolbox.register("selector", create_units, self.selector)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.selector)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("winner", tools.selBest)

        return toolbox

    def _run_gen(self, pop):
        # Select individuals for the next generation
        offspring = self.toolbox.select(pop, len(pop))

        # Clone the selected individuals
        offspring = list(map(self.toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.cxpb:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < self.mutpb:
                for _ in range(self.mutnb):
                    self.toolbox.mutate(mutant, self.selector)
                del mutant.fitness.values

        # Evaluate mutated or crossed individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        formatted_invalid = map(format_ind, invalid_ind)
        fitnesses = map(self.toolbox.evaluate, formatted_invalid)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("Generation Finished!")
        return offspring


class FullSearch(Search):
    def __init__(self, network, state, selector, max_tries):
        super().__init__(network, state)
        self.selector = selector
        self.states = []
        self.max_tries = max_tries
        self.tries = 0

    def search(self):
        while self.tries < self.max_tries:
            state = self.selector.select()
            if state in self.states:
                self.tries += 1
            else:
                self.states.append(state)
                self.tries = 0

        max_wp, best_ally = 0.0, None
        for state in self.states:
            (wp,) = self.classifier.eval(state)
            if wp > max_wp:
                max_wp, best_ally = wp, state
        return best_ally, max_wp


class MonteCarloSearch(Search):
    def __init__(self, network, state, selector, n_states):
        super().__init__(network, state)
        self.n_states = n_states
        self.selector = selector

    def search(self):
        max_wp, best_ally = 0.0, None
        for _ in range(self.n_states):
            ally = self.selector.select()
            (wp,) = self.classifier.eval(ally)
            if wp > max_wp:
                max_wp = wp
                best_ally = ally

        return best_ally, max_wp
