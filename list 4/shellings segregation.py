import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from matplotlib import animation
import itertools
import copy
import multiprocessing as mp
import datetime


class ShellingsSegregation:
    # availible kind of agents
    EMPTY = 0
    BLUE = 1
    RED = 2

    # class agent with parameters: kind and localization
    class Agent:
        def __init__(self, type, x_loc, y_loc):
            self.type = type
            self.x_loc = x_loc
            self.y_loc = y_loc

    # constructor for simulation
    def __init__(self, blue_agents_number, red_agents_number, m_t, j_t, lattice_size=100, j_b=None, j_r=None):
        self.m_t = m_t

        # case when there is only one parameter j for both classes
        if j_r is not None and j_r is not None:
            self.j_r = j_r
            self.j_b = j_b
            self.j_t = None
        # case when there different j for both classes
        else:
            self.j_t = j_t
            self.j_r = None
            self.j_b = None
        self.lattice_size = lattice_size
        # init of lattice which is matrix LxL with agents as value
        self.lattice = np.empty(shape=(self.lattice_size, self.lattice_size), dtype=self.Agent)

        # at the beggining lattice is EMPTY
        for i in range(self.lattice_size):
            for j in range(self.lattice_size):
                self.lattice[i][j] = self.Agent(self.EMPTY, i, j)

        # init of needed lists
        self.agents = []
        self.agents_at_the_start = None
        self.frame_no = 0
        self.unhappy_agents_before_move_in_each_step = []
        self.unhappy_agents_after_move_in_each_step = []

        # generating randolmy busy location at the begginning
        self.busy_locations = self.generate_starting_locations(blue_agents_number, red_agents_number)
        # initialization of agents of both classes
        self.init_agents(blue_agents_number, red_agents_number)

    def generate_starting_locations(self, blue_agents_number, red_agents_number):
        coordinates = np.arange(0, self.lattice_size)
        # we generate all possible coordinates
        all_possible_coordinates = list(itertools.product(coordinates, repeat=2))
        # we mix all possible agents and return fhe first n, n is a number of all agents
        np.random.shuffle(all_possible_coordinates)
        return all_possible_coordinates[:blue_agents_number + red_agents_number]

    def init_agents(self, blue_agents_number, red_agents_number):

        blue_agents_starting_location = self.busy_locations[:blue_agents_number]
        red_agents_starting_location = self.busy_locations[blue_agents_number:]

        # creating blue agents
        for x, y in blue_agents_starting_location:
            new_agent = ShellingsSegregation.Agent(ShellingsSegregation.BLUE, x, y)
            self.add_new_agent(new_agent)
            self.lattice[x][y] = new_agent

        # creating red agents
        for x, y in red_agents_starting_location:
            new_agent = ShellingsSegregation.Agent(ShellingsSegregation.RED, x, y)
            self.add_new_agent(new_agent)
            self.lattice[x][y] = new_agent

    # adding agents to list of agents
    def add_new_agent(self, new_agent):
        self.agents.append(new_agent)

    # unused method, it was used in the first implementation to calculate euclidain distance between agents
    def distance_between(self, given_agent, other_agent):
        x_difference = abs(given_agent.x_loc - other_agent.x_loc)
        if x_difference > self.lattice_size / 2:
            x_difference = self.lattice_size - x_difference

        y_difference = abs(given_agent.y_loc - other_agent.y_loc)
        if y_difference > self.lattice_size / 2:
            y_difference = self.lattice_size - y_difference

        return np.sqrt(x_difference * x_difference + y_difference * y_difference)

    # method which perform simulation
    def run_simulation(self, draw_plots=True, animate=False):

        print("Start " + str(datetime.datetime.now().time()))
        print("Population size: " + str(len(self.agents) / 2))
        cycle_no = 0
        any_moved = True
        if draw_plots:
            self.draw_lattice(cycle_no)
        if animate:
            self.agents_at_the_start = copy.deepcopy(self.agents)

        # step is performed if there was any moved agent in the previous one
        while any_moved:
            cycle_no = cycle_no + 1
            any_moved = False
            moved_agents_number = 0

            if animate:
                moved_agents_before_move = []
                moved_agents_after_move = []

            # we iterate through all agents
            for agent in self.agents:

                # finding the m closest agents of the agent
                closest_agents = self.find_m_t_closest_agents_for(agent)
                # determining the number of neighbours with the same type
                the_same_type_neighbours_no = self.the_same_type_neighbours_number(agent, closest_agents)


                move_agent = False
                # the case when j_t is the same for both classes
                if self.j_t is not None:
                    # we move agent when tha ration of the same kind neighbours to all neighbours is less than j_t
                    move_agent = the_same_type_neighbours_no / self.m_t < self.j_t
                # the case when the agent is BLUE - we take the threshold j_b for BLUE agents
                elif agent.type == self.BLUE:
                    move_agent = the_same_type_neighbours_no / self.m_t < self.j_b
                # the case when the agent is BLUE - we take the threshold j_r for BLUE agents
                else:
                    move_agent = the_same_type_neighbours_no / self.m_t < self.j_r

                # we move agent
                if move_agent:
                    # setting that there is any agent moved in the step
                    any_moved = True

                    moved_agents_number = moved_agents_number + 1

                    if animate:
                        # coping the agent before a move - needed during animation
                        moved_agents_before_move.append(copy.copy(agent))

                    # calling method which moves the agent
                    self.move_agent_to_empty_place(agent)

                    if animate:
                        # coping the agent after a move - needed during animation
                        moved_agents_after_move.append(copy.copy(agent))

            print("Step " + str(cycle_no) + ', moved agents: ' + str(moved_agents_number))
            # drawing each 10th step
            if draw_plots and cycle_no % 10 == 0:
                self.draw_lattice(cycle_no)
            if animate:
                self.unhappy_agents_before_move_in_each_step.append(moved_agents_before_move)
                self.unhappy_agents_after_move_in_each_step.append(moved_agents_after_move)

        print("End")

        if draw_plots:
            self.draw_lattice(cycle_no)
        if animate:
            self.frame_no = cycle_no+1
            self.animate_simulation()

        # detemining segregation index
        segregation_index = self.calculate_similar_neighbour_index()

        # method returns segragtion index and number of steps
        return segregation_index, cycle_no

    def find_m_t_closest_agents_for(self, agent):
        closest_agents = []
        agent_x = agent.x_loc
        agent_y = agent.y_loc
        max_delta = 1  # the max cell gap between the agent and the seeking neighbour

        # we seeks for neighbour until there is m_t found neighbours
        while len(closest_agents) < self.m_t:
            # possible !D distance between agent and neighbours
            deltas = np.arange(-max_delta, max_delta + 1)

            # we take all possible cell around the agent with determined max_delta
            for delta_x in deltas:
                for delta_y in deltas:
                    # we use modulo becaiuse the lattice is to be a torus
                    other_agent = self.lattice[(agent_x + delta_x) % self.lattice_size][
                        (agent_y + delta_y) % self.lattice_size]
                    # we add an other agent if is not EMPTY and if the neighbour is not yet added
                    if other_agent.type != self.EMPTY and other_agent not in closest_agents and other_agent != agent:
                        closest_agents.append(other_agent)

            # we increas the are of searching
            max_delta = max_delta + 1

        # we take m_t closes agents
        return closest_agents[:self.m_t]


    def the_same_type_neighbours_number(self, agent, closest_agents):
        the_same_type_neighbours_no = 0  # accumulator which measure the closes neighbours with the same type
        for closest_agent in closest_agents:
            # if the same type increas accumulator
            if closest_agent.type == agent.type:
                the_same_type_neighbours_no = the_same_type_neighbours_no + 1
        return the_same_type_neighbours_no

    def move_agent_to_empty_place(self, agent):
        old_x_loc = agent.x_loc
        old_y_loc = agent.y_loc


        new_x_loc, new_y_loc = self.generate_random_localization()

        # we generate new localization until its cell is empty
        while (new_x_loc, new_y_loc) in self.busy_locations:
            new_x_loc, new_y_loc = self.generate_random_localization()

        # we find the indes of localization of our agent in busy_localization list
        agent_loc_ind = np.where(self.busy_locations == (old_x_loc, old_y_loc))
        # deleting thep previous agent's localization form the list
        self.busy_locations = np.delete(self.busy_locations, agent_loc_ind, axis=0)
        # we set EMPTY agnet in the previous localization on the lattice
        self.lattice[old_x_loc][old_y_loc] = self.Agent(self.EMPTY, old_x_loc, old_y_loc)

        # we set new localization for the agent
        agent.x_loc = new_x_loc
        agent.y_loc = new_y_loc

        # we add new localization to the buy_localization list
        self.busy_locations = np.append(self.busy_locations, [(new_x_loc, new_y_loc)])
        # we set the agent in the new localization on the lattice
        self.lattice[new_x_loc][new_y_loc] = agent

    def generate_random_localization(self):
        result = np.random.randint(self.lattice_size), np.random.randint(self.lattice_size)
        return result

    def calculate_similar_neighbour_index(self):
        proportions = []
        # calculating the ratio of the same type neighbours to the all neighbours for each agent
        for agent in self.agents:
            closest_agents = self.find_m_t_closest_agents_for(agent)
            the_same_type_neighbours_no = self.the_same_type_neighbours_number(agent, closest_agents)

            proportions.append(the_same_type_neighbours_no / len(closest_agents))
        # take average of all ratios, this is segragetion index
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

        def animate(frame):
            patches_to_return = []
            if frame == 0:
                agents_to_draw = self.agents_at_the_start
                for agent in agents_to_draw:
                    color = 'blue'
                    if agent.type == ShellingsSegregation.RED:
                        color = 'red'
                    rectangle = Rectangle((agent.y_loc, self.lattice_size - agent.x_loc - 1), 1, 1, facecolor=color,
                                          alpha=1)
                    ax.add_patch(rectangle)
                    patches_to_return.append(rectangle)
            if frame >= 1:

                agents_before_move = self.unhappy_agents_before_move_in_each_step[frame-1]
                for agent in agents_before_move:
                    color = 'white'
                    rectangle = Rectangle((agent.y_loc, self.lattice_size - agent.x_loc - 1), 1, 1, facecolor=color,
                                          alpha=1)
                    ax.add_patch(rectangle)
                    patches_to_return.append(rectangle)

                agents_after_move = self.unhappy_agents_after_move_in_each_step[frame-1]
                for agent in agents_after_move:
                    color = 'blue'
                    if agent.type == ShellingsSegregation.RED:
                        color = 'red'
                    rectangle = Rectangle((agent.y_loc, self.lattice_size - agent.x_loc - 1), 1, 1, facecolor=color,
                                          alpha=1)
                    ax.add_patch(rectangle)
                    patches_to_return.append(rectangle)

            ax.set_title("Cycle no: " + str(frame))
            print("Frame " + str(frame) + " animated")
            return patches_to_return

        fig, ax = plt.subplots()
        frame_amount = self.frame_no
        interval_time = 300
        anim = animation.FuncAnimation(fig, animate,
                                       init_func=init_animation,
                                       frames=frame_amount,
                                       interval=interval_time,
                                       blit=True,
                                       repeat=False)

        if self.j_t is not None:
            gif_title = "2Segregation pop=" + str(len(self.agents) / 2) + "L=" + str(self.lattice_size) + " m_t=" + str(
                self.m_t) + " j_t=" + str(self.j_t)
        else:
            gif_title = "2Segregation pop=" + str(len(self.agents) / 2) + "L=" + str(self.lattice_size) + " m_t=" + str(
                self.m_t) + " j_b=" + str(self.j_b) + " j_r=" + str(self.j_r)
        anim.save(gif_title + '.gif', writer='imagemagick')
        # plt.show()


def task_one_and_two():
    shellings_segregation = ShellingsSegregation(4000, 4000, 8, 0.5, 100)
    segregation_ind, cycle_no = shellings_segregation.run_simulation(True, False)
    print('Segregation simulation:\n' +
          ' - cycles number: ' + str(cycle_no) + '\n' +
          ' - segregation index: ' + str(segregation_ind)
          )


def iteration_number_of_simulation(m_t, j_t, L, pop_size):
    iteretions_numbers = []
    # we make n simulations to take average iteration index for each pop size
    n = 5;
    for i in range(n):
        segregation = ShellingsSegregation(pop_size, pop_size, m_t, j_t, L)
        _, iterations_no = segregation.run_simulation(draw_plots=False)
        iteretions_numbers.append(iterations_no)
    return np.mean(iterations_no)


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
    file_title = 'Iteration plot2 ' + 'm_t=' + str(m_t) + ' j_t=' + str(j_t) + ' L=' + str(L) + '.png'
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
        j_b=3 / 8,
        j_r=6 / 8,
        j_t=None
    )
    segregation_ind, cycle_no = shellings_segregation.run_simulation(True, False)
    print('Segregation simulation:\n' +
          ' - cycles number: ' + str(cycle_no) + '\n' +
          ' - segregation index: ' + str(segregation_ind)
          )


if __name__ == "__main__":
    # task_one_and_two()
    # task_three()
    # task_four()
    # task_five()
    task_six()
