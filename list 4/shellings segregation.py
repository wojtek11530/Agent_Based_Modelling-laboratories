import numpy as np
import matplotlib.pyplot as plt
from anaconda_project.internal.test.fields import six
from matplotlib.patches import Rectangle
from matplotlib import animation
import itertools
import copy
import multiprocessing as mp
import datetime


class ShellingsSegregation:
    EMPTY = 0
    BLUE = 1
    RED = 2

    class Agent:
        def __init__(self, type, x_loc, y_loc):
            self.type = type
            self.x_loc = x_loc
            self.y_loc = y_loc

    def __init__(self, blue_agents_number, red_agents_number, m_t, j_t, lattice_size=100, j_b=None, j_r=None):
        self.m_t = m_t
        if j_r is not None and j_r is not None:
            self.j_r = j_r
            self.j_b = j_b
            self.j_t = None
        else:
            self.j_t = j_t
            self.j_r = None
            self.j_b = None
        self.lattice_size = lattice_size
        self.lattice = np.empty(shape=(self.lattice_size, self.lattice_size), dtype=self.Agent)

        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                self.lattice[i][j] = self.Agent(self.EMPTY, i, j)

        self.agents = []
        self.agents_in_each_step = []

        self.busy_locations = self.generate_starting_locations(blue_agents_number, red_agents_number)
        self.init_agents(blue_agents_number, red_agents_number)

    def generate_starting_locations(self, blue_agents_number, red_agents_number):
        coordinates = np.arange(0, self.lattice_size)
        all_possible_coordinates = list(itertools.product(coordinates, repeat=2))
        np.random.shuffle(all_possible_coordinates)
        return all_possible_coordinates[:blue_agents_number + red_agents_number]

    def init_agents(self, blue_agents_number, red_agents_number):

        blue_agents_starting_location = self.busy_locations[:blue_agents_number]
        red_agents_starting_location = self.busy_locations[blue_agents_number:]

        for x, y in blue_agents_starting_location:
            new_agent = ShellingsSegregation.Agent(ShellingsSegregation.BLUE, x, y)
            self.add_new_agent(new_agent)
            self.lattice[x][y] = new_agent

        for x, y in red_agents_starting_location:
            new_agent = ShellingsSegregation.Agent(ShellingsSegregation.RED, x, y)
            self.add_new_agent(new_agent)
            self.lattice[x][y] = new_agent

    def add_new_agent(self, new_agent):
        self.agents.append(new_agent)

    def distance_between(self, given_agent, other_agent):
        x_difference = abs(given_agent.x_loc - other_agent.x_loc)
        if x_difference > self.lattice_size / 2:
            x_difference = self.lattice_size - x_difference

        y_difference = abs(given_agent.y_loc - other_agent.y_loc)
        if y_difference > self.lattice_size / 2:
            y_difference = self.lattice_size - y_difference

        return np.sqrt(x_difference * x_difference + y_difference * y_difference)

    def run_simulation(self, draw_plots=True, animate=False):

        print("Start " + str(datetime.datetime.now().time()))
        print("Population size: " + str(len(self.agents) / 2))
        cycle_no = 0
        any_moved = True
        if draw_plots:
            self.draw_lattice(cycle_no)
        if animate:
            self.agents_in_each_step.append(np.copy(self.agents))

        while any_moved:
            cycle_no = cycle_no + 1
            any_moved = False
            moved_agents = 0
            for agent in self.agents:
                closest_agents = self.find_m_t_closest_agents_for(agent)
                the_same_type_neighbours_no = self.the_same_type_neighbours_number(agent, closest_agents)
                if self.j_t is not None:
                    if the_same_type_neighbours_no / self.m_t < self.j_t:
                        any_moved = True
                        moved_agents = moved_agents + 1
                        self.move_agent_to_empty_place(agent)
                elif agent.type == self.BLUE:
                    if the_same_type_neighbours_no / self.m_t < self.j_b:
                        any_moved = True
                        moved_agents = moved_agents + 1
                        self.move_agent_to_empty_place(agent)
                else:
                    if the_same_type_neighbours_no / self.m_t < self.j_r:
                        any_moved = True
                        moved_agents = moved_agents + 1
                        self.move_agent_to_empty_place(agent)

            print("Step " + str(cycle_no) + ', moved agents: ' + str(moved_agents))
            if draw_plots and cycle_no % 10 == 0:
                self.draw_lattice(cycle_no)
            if animate:
                copy_agents = copy.deepcopy(self.agents)
                self.agents_in_each_step.append(copy_agents)

        print("End")
        if draw_plots:
            self.draw_lattice(cycle_no)
        if animate:
            self.animate_simulation()

        segregation_index = self.calculate_similar_neighbour_index()
        return segregation_index, cycle_no

    def find_m_t_closest_agents_for(self, agent):
        closest_agents = []
        agent_x = agent.x_loc
        agent_y = agent.y_loc
        max_delta = 1
        while len(closest_agents) < self.m_t:
            deltas = np.arange(-max_delta, max_delta + 1)
            for delta_x in deltas:
                for delta_y in deltas:
                    other_agent = self.lattice[(agent_x + delta_x) % self.lattice_size][
                        (agent_y + delta_y) % self.lattice_size]
                    if other_agent.type != self.EMPTY and other_agent not in closest_agents and other_agent != agent:
                        closest_agents.append(other_agent)

            max_delta = max_delta + 1

        return closest_agents[:self.m_t]

    def the_same_type_neighbours_number(self, agent, closest_agents):
        the_same_type_neighbours_no = 0
        for closest_agent in closest_agents:
            if closest_agent.type == agent.type:
                the_same_type_neighbours_no = the_same_type_neighbours_no + 1
        return the_same_type_neighbours_no

    def move_agent_to_empty_place(self, agent):
        old_x_loc = agent.x_loc
        old_y_loc = agent.y_loc

        new_x_loc, new_y_loc = self.generate_random_localization()

        while (new_x_loc, new_y_loc) in self.busy_locations:
            new_x_loc, new_y_loc = self.generate_random_localization()

        agent_loc_ind = np.where(self.busy_locations == (old_x_loc, old_y_loc))
        self.busy_locations = np.delete(self.busy_locations, agent_loc_ind, axis=0)
        self.lattice[old_x_loc][old_y_loc] = self.Agent(self.EMPTY, old_x_loc, old_y_loc)

        agent.x_loc = new_x_loc
        agent.y_loc = new_y_loc
        self.busy_locations = np.append(self.busy_locations, [(new_x_loc, new_y_loc)])
        self.lattice[new_x_loc][new_y_loc] = agent

    def generate_random_localization(self):
        result = np.random.randint(self.lattice_size), np.random.randint(self.lattice_size)
        return result

    def calculate_similar_neighbour_index(self):
        proportions = []
        for agent in self.agents:
            closest_agents = self.find_m_t_closest_agents_for(agent)
            the_same_type_neighbours_no = self.the_same_type_neighbours_number(agent, closest_agents)
            proportions.append(the_same_type_neighbours_no / len(closest_agents))

        return np.mean(proportions)

    def draw_lattice(self, cycle_no=None):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.lattice_size)
        ax.set_ylim(0, self.lattice_size)
        ax.set_aspect('equal')

        for agent in self.agents:
            color = 'blue'
            if agent.type == ShellingsSegregation.RED:
                color = 'red'
            rectangle = Rectangle((agent.y_loc, self.lattice_size - agent.x_loc - 1), 1, 1, facecolor=color, alpha=1)
            ax.add_patch(rectangle)

        if cycle_no is not None:
            plt.title("Cycle no: " + str(cycle_no))
        plt.show()

    def animate_simulation(self):
        # Function preparing an animation
        def init_animation():
            # Setting axis parameters and title
            ax.set_xlim(0, self.lattice_size)
            ax.set_ylim(0, self.lattice_size)
            ax.set_aspect('equal')
            ax.set_axis_off()
            patches_to_return = []
            return patches_to_return

        def animate(step):
            patches_to_return = []
            agents_in_step = self.agents_in_each_step[step]
            if step >= 1:
                agents_in_prev_step = self.agents_in_each_step[step - 1]
                for agent in agents_in_prev_step:
                    color = 'white'
                    rectangle = Rectangle((agent.y_loc, self.lattice_size - agent.x_loc - 1), 1, 1, facecolor=color,
                                          alpha=1)
                    ax.add_patch(rectangle)
                    patches_to_return.append(rectangle)

            for agent in agents_in_step:
                color = 'blue'
                if agent.type == ShellingsSegregation.RED:
                    color = 'red'
                rectangle = Rectangle((agent.y_loc, self.lattice_size - agent.x_loc - 1), 1, 1, facecolor=color,
                                      alpha=1)
                ax.add_patch(rectangle)
                patches_to_return.append(rectangle)

            ax.set_title("Cycle no: " + str(step))
            print("Step " + str(step) + " animated")
            return patches_to_return

        fig, ax = plt.subplots()
        frame_amount = len(self.agents_in_each_step)
        interval_time = 300
        anim = animation.FuncAnimation(fig, animate,
                                       init_func=init_animation,
                                       frames=frame_amount,
                                       interval=interval_time,
                                       blit=True,
                                       repeat=False)

        if self.j_t is not None:
            gif_title = "Segregation pop=" + str(len(self.agents) / 2) + "L=" + str(self.lattice_size) + " m_t=" + str(
                self.m_t) + " j_t=" + str(self.j_t)
        else:
            gif_title = "Segregation pop=" + str(len(self.agents) / 2) + "L=" + str(self.lattice_size) + " m_t=" + str(
                self.m_t) + " j_b=" + str(self.j_b) + " j_r=" + str(self.j_r)
        anim.save(gif_title + '.gif', writer='imagemagick')
        # plt.show()


def task_one_and_two():
    shellings_segregation = ShellingsSegregation(4000, 4000, 8, 0.5, 100)
    segregation_ind, cycle_no = shellings_segregation.run_simulation(True, True)
    print('Segregation simulation:\n' +
          ' - cycles number: ' + str(cycle_no) + '\n' +
          ' - segregation index: ' + str(segregation_ind)
          )


def iteration_number_of_simulation(m_t, j_t, L, pop_size):
    segregation = ShellingsSegregation(pop_size, pop_size, m_t, j_t, L)
    _, iterations_no = segregation.run_simulation(draw_plots=False)
    return iterations_no


def task_three():
    pool = mp.Pool(mp.cpu_count())
    m_t = 8
    j_t = 0.5
    L = 100
    populations = np.arange(250, 4000, 250)
    iterations_numbers = pool.starmap(iteration_number_of_simulation,
                                      [(m_t, j_t, L, pop_size) for pop_size in populations])
    save_iterations_plot(iterations_numbers, populations, m_t, j_t, L)


def save_iterations_plot(iterations_numbers, populations, m_t, j_t, L):
    plt.plot(populations, iterations_numbers, '.')
    plt.xlabel('population of each group')
    plt.ylabel('iterations')
    plt.title(r'Number of iterations, $m_t=$' + str(m_t) + r', $j_t=$' + str(
        j_t) + r', $L=$' + str(L))
    plt.grid()
    file_title = 'Iteration plot ' + 'm_t=' + str(m_t) + ' j_t=' + str(j_t) + ' L=' + str(L) + '.png'
    plt.savefig(file_title)


def get_segregation_index_of_simulation(m_t, j_t, L, pop_size):
    segregation = ShellingsSegregation(pop_size, pop_size, m_t, j_t, L)
    segregation_index, _ = segregation.run_simulation(draw_plots=False)
    return segregation_index


def task_four():
    pool = mp.Pool(mp.cpu_count())
    m_t = 10
    j_ts = np.arange(0.1, 1, 0.1)
    L = 100
    population = 250
    segregation_indexes = pool.starmap(get_segregation_index_of_simulation,
                                       [(m_t, j_t, L, population) for j_t in j_ts])
    save_segregation_ind_to_j_t_plot(j_ts, segregation_indexes, m_t, L, population)


def save_segregation_ind_to_j_t_plot(j_ts, segregation_indexes, m_t, L, population):
    plt.plot(j_ts, segregation_indexes, '.')
    plt.xlabel(r'$j_t$')
    plt.ylabel('segregation index')
    plt.title(r'Segregation indexes, $m_t=$' + str(m_t) + r', pop_size=' + str(
        population) + r', $L=$' + str(L))
    plt.grid()
    file_title = 'Segregation plot ' + 'm_t=' + str(m_t) + ' pop_size=' + str(population) + ' L=' + str(L) + '.png'
    plt.savefig(file_title)


def task_five():
    pool = mp.Pool(mp.cpu_count())
    m_ts = [8, 24, 48, 80, 120]
    j_t = 0.5
    L = 100
    population = 250
    segregation_indexes = pool.starmap(get_segregation_index_of_simulation,
                                       [(m_t, j_t, L, population) for m_t in m_ts])
    save_segregation_ind_to_m_t_plot(m_ts, segregation_indexes, j_t, L, population)


def save_segregation_ind_to_m_t_plot(m_ts, segregation_indexes, j_t, L, population):
    plt.plot(m_ts, segregation_indexes, '.')
    plt.xlabel(r'$m_t$')
    plt.ylabel('segregation index')
    plt.title(r'Segregation indexes, $j_t=$' + str(j_t) + r', pop_size=' + str(
        population) + r', $L=$' + str(L))
    plt.grid()
    file_title = 'Segregation plot ' + 'j_t=' + str(j_t) + ' pop_size=' + str(population) + ' L=' + str(L) + '.png'
    plt.savefig(file_title)


def task_six():
    shellings_segregation = ShellingsSegregation(
        blue_agents_number=250,
        red_agents_number=250,
        m_t=8,
        lattice_size=100,
        j_b=3/8,
        j_r=6/8,
        j_t=None
    )
    segregation_ind, cycle_no = shellings_segregation.run_simulation(True, True)
    print('Segregation simulation:\n' +
          ' - cycles number: ' + str(cycle_no) + '\n' +
          ' - segregation index: ' + str(segregation_ind)
          )


if __name__ == "__main__":
    # task_one()
    # task_three()
    # task_four()
    # task_five()
    task_six()
