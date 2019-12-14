import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation

# three possible state of cells
EMPTY = 1
TREE = 2
BURNING = 3
BURNED_TREE = 4


def perform_some_forest_fire_simulations():
    global prob_of_tree, lattice_size
    prob_of_tree = 0.45
    lattice_sizes = (20, 50, 100)
    simulation_animated = True

    # For three size of lattice perform a simulation of fire
    for lattice_size in lattice_sizes:
        make_simulation_of_fire_in_forest(prob_of_tree, lattice_size, simulation_animated)


def make_simulation_of_fire_in_forest(prob_of_tree, lattice_size, simulation_animated=False, HK_algorihm=False):
    lattices_in_each_step = []  # A list containing a state of the lattice for each step of simulation
    lattice = init_lattice(lattice_size, prob_of_tree)  # A function initializing a lattice

    opposite_edge_on_fire, size_of_biggest_cluster = run_simulation(lattice_size, lattice, lattices_in_each_step, HK_algorihm)

    if simulation_animated:
        animate_simulation(lattices_in_each_step, lattice_size)

    # Returning of the results
    return opposite_edge_on_fire, size_of_biggest_cluster


def init_lattice(lattice_size, prob_of_tree):
    lattice = np.empty([lattice_size, lattice_size])
    put_trees_randomly(lattice, prob_of_tree, lattice_size)
    set_fire_on_the_left_edge(lattice, lattice_size)
    return lattice


def put_trees_randomly(lattice, prob_of_tree, lattice_size):
    for i in range(0, lattice_size):
        for j in range(0, lattice_size):
            # If random number from (0,1) is lower than probability then a tree is put in the cell
            # In opposite case the cell is left EMPTY
            if np.random.rand() < prob_of_tree:
                lattice[i][j] = TREE
            else:
                lattice[i][j] = EMPTY


def set_fire_on_the_left_edge(lattice, lattice_size):
    # Set fire on the left edge which means on the 0th column of the array
    for i in range(0, lattice_size):
        lattice[i][0] = BURNING


# A function performing a simulation. It  returns boolean which indicates if the opposite edge was on fire
# and, if HK_algorithm is true, the size of the biggest cluster
def run_simulation(lattice_size, init_lattice, lattices_in_each_step, HK_algorihm=False):
    current_lattice = np.copy(init_lattice)
    lattices_in_each_step.append(init_lattice)

    opposite_edge_on_fire = False
    max_cluster_size = None

    if HK_algorihm:
        labels = perform_Hoshen_Kopelman_algorithm(current_lattice)  # performing HK algorithm which return labels array
        max_cluster_size = find_max_cluster_size(labels)  # finding the biggest size of BURNED_TREE  cluster

    while any_fire_on_lattice(current_lattice, lattice_size):  # simulation lasts until there is any fire on the grid
        new_lattice = np.copy(current_lattice)
        for i in range(0, lattice_size):
            for j in range(0, lattice_size):
                # BURNING cell turns into BURNED_TREE one
                if current_lattice[i][j] == BURNING:
                    new_lattice[i][j] = BURNED_TREE
                # a cell with TREE and with BURNING neighbour becomes BURNING one
                elif current_lattice[i][j] == TREE and \
                        any_neighbour_of_cell_is_burning(current_lattice, lattice_size, i, j):
                    new_lattice[i][j] = BURNING
                    # Checking if the column number of the just set on fire cell is the last column number
                    # if yes it means that the opposite edge is on fire
                    if j == lattice_size - 1:
                        opposite_edge_on_fire = True

        if HK_algorihm:
            labels = perform_Hoshen_Kopelman_algorithm(new_lattice)
            current_max_cluster_size = find_max_cluster_size(labels)
            max_cluster_size = max(current_max_cluster_size, max_cluster_size)

        lattices_in_each_step.append(new_lattice)
        current_lattice = new_lattice

    return opposite_edge_on_fire, max_cluster_size


def any_fire_on_lattice(lattice, lattice_size):
    for i in range(0, lattice_size):
        for j in range(0, lattice_size):
            if lattice[i][j] == BURNING:
                return True
    return False


# Checking if there exists any neighbour of the cell (i,j) which is on fire.
# We have to be sure that we are not out of bounds of the array.
def any_neighbour_of_cell_is_burning(lattice, lattice_size, i, j):
    if i - 1 >= 0 and j - 1 >= 0 and lattice[i - 1][j - 1] == BURNING:
        return True
    elif i - 1 >= 0 and lattice[i - 1][j] == BURNING:
        return True
    elif i - 1 >= 0 and j + 1 < lattice_size and lattice[i - 1][j + 1] == BURNING:
        return True
    elif j - 1 >= 0 and lattice[i][j - 1] == BURNING:
        return True
    elif j + 1 < lattice_size and lattice[i][j + 1] == BURNING:
        return True
    elif i + 1 < lattice_size and j - 1 >= 0 and lattice[i + 1][j - 1] == BURNING:
        return True
    elif i + 1 < lattice_size and lattice[i + 1][j] == BURNING:
        return True
    elif i + 1 < lattice_size and j + 1 < lattice_size and lattice[i + 1][j + 1] == BURNING:
        return True
    else:
        return False


def draw_lattice(lattice):
    plt.figure()
    current_axis = plt.gca()
    for i in range(0, lattice_size):
        for j in range(0, lattice_size):
            color = 'yellow'
            if lattice[i][j] == TREE:
                color = 'green'
            elif lattice[i][j] == BURNING:
                color = 'red'
            elif lattice[i][j] == BURNED_TREE:
                color = 'brown'

            # Rectangle(xy, width, height, angle=0.0, **kwargs)
            # A rectangle with lower left at xy = (x, y) with specified width, height and rotation angle.
            # (x, y) is the the bottom and left rectangle coordinates
            # so we have to transform row and column of array into xy-coordinate. Column number j is x-coordinate
            # and the lattice_size - row_number - 1 is y coordinate
            rectangle = Rectangle((j, lattice_size - i - 1), 1, 1, facecolor=color, edgecolor='black', alpha=1)
            current_axis.add_patch(rectangle)
    plt.xlim([0, lattice_size])
    plt.ylim([0, lattice_size])
    plt.axis('equal')
    plt.show()


# Function drawing an animation of the simulation, it needs state of the lattice at the each step
def animate_simulation(lattices_in_each_step, lattice_size):

    # Function preparing an animation
    def init_animation():

        # Setting axis parameters and title
        ax.set_xlim(0, lattice_size)
        ax.set_ylim(0, lattice_size)
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title("latticeSize = " + str(lattice_size) + " treeProb = " + str(prob_of_tree))

        patches_to_return = []
        for i in range(0, lattice_size):
            for j in range(0, lattice_size):
                rectangle = patches[i][j]
                # Adding to the plot rectangles
                ax.add_patch(rectangle)
                patches_to_return.append(rectangle)
        return patches_to_return

    def animate(step):

        patches_to_return = []

        lattice_to_draw = lattices_in_each_step[step]  # The state of lattice at the given step
        for i in range(0, lattice_size):
            for j in range(0, lattice_size):
                color = 'yellow'  # default color
                if lattice_to_draw[i][j] == TREE:
                    color = 'green'
                elif lattice_to_draw[i][j] == BURNING:
                    color = 'red'
                elif lattice_to_draw[i][j] == BURNED_TREE:
                    color = 'brown'

                rectangle: Rectangle = patches[i][j]
                rectangle.set_facecolor(color)  # changing color of the rectangle
                patches_to_return.append(rectangle)

        return patches_to_return

    fig, ax = plt.subplots()
    patches = []  # list of being drawn patches (rectangles)

    for i in range(0, lattice_size):
        patches_in_row = []
        for j in range(0, lattice_size):

            # We initilize rectangles representing each cell of lattice
            rectangle = Rectangle((j, lattice_size - i - 1), 1, 1, edgecolor='black', alpha=1)
            patches_in_row.append(rectangle)
        patches.append(patches_in_row)

    frame_amount = len(lattices_in_each_step)
    interval_time = 50
    anim = animation.FuncAnimation(fig, animate,
                                   init_func=init_animation,
                                   frames=frame_amount,
                                   interval=interval_time,
                                   blit=True,
                                   repeat=False)
    gif_title = "Fire in forest L = " + str(lattice_size) + " p = " + str(prob_of_tree)
    anim.save(gif_title + '.gif', writer='imagemagick')
    # plt.show()


def determine_percolation_threshold_for_various_lattice_sizes():
    simulation_no = 50
    tree_probabilities = np.arange(0.1, 1, 0.05)
    lattice_sizes = (20, 50, 100)
    for lattice_size in lattice_sizes:
        percolation_thresholds = []  # list of threshold results
        for tree_prob in tree_probabilities:
            percolation_thresholds.append(
                determine_percolation_threshold(lattice_size, tree_prob, simulation_no))

        save_percolation_plot(lattice_size, simulation_no, tree_probabilities, percolation_thresholds)


# Function calculating percolation threshold for simulation run $simulation_no times
def determine_percolation_threshold(lattice_size, tree_prob, simulation_no):
    opposite_edge_on_fire_count = 0
    for n in range(0, simulation_no):
        opp_edge_on_fire, _ = make_simulation_of_fire_in_forest(tree_prob, lattice_size)

        # Increasing number of simulation when fire reached the opposite edge
        if opp_edge_on_fire:
            opposite_edge_on_fire_count = opposite_edge_on_fire_count + 1

    # Average number of times when fire reached the opposite edge
    percolation_threshold = opposite_edge_on_fire_count / simulation_no
    return percolation_threshold


def save_percolation_plot(lattice_size, simulation_no, tree_probabilities, percolation_thresholds):
    fig, ax = plt.subplots()
    ax.plot(tree_probabilities, percolation_thresholds)
    ax.set_title(r'Percolation threshold for $L=$' + str(lattice_size) + r', simulation_amount=' + str(simulation_no))
    ax.set_xlabel(r'Tree probabilities $p$')
    ax.set_ylabel("Percolation threshold")
    ax.grid()
    file_title = 'Percolation threshold plot, L=' + str(lattice_size) + 'sim_no=' + str(simulation_no) + '.png'
    plt.savefig(file_title)


def perform_Hoshen_Kopelman_algorithm(lattice):
    largest_label = 0
    lattice_size = len(lattice)
    labels = np.zeros((lattice_size, lattice_size))
    for j in range(0, len(lattice)):
        for i in range(0, len(lattice)):
            if lattice[i][j] == BURNED_TREE:
                is_left_burned = False
                if j - 1 >= 0 and lattice[i][j - 1] == BURNED_TREE:
                    is_left_burned = True
                is_above_burned = False
                if i - 1 >= 0 and lattice[i - 1][j] == BURNED_TREE:
                    is_above_burned = True
                if not is_left_burned and not is_above_burned:
                    largest_label = largest_label + 1
                    labels[i, j] = largest_label
                elif is_left_burned and not is_above_burned:
                    labels[i, j] = labels[i][j - 1]
                elif is_above_burned and not is_left_burned:
                    labels[i, j] = labels[i - 1][j]
                else:
                    left_label = labels[i][j - 1]
                    above_label = labels[i - 1][j]
                    union(labels, left_label, above_label)
                    labels[i, j] = above_label
    return labels


def union(labels, left_label, above_label):
    for label_rows in labels:
        for label in label_rows:
            if label == left_label:
                label = above_label


def find_max_cluster_size(labels):
    labels_amount_dict = {}
    for label_rows in labels:
        for label in label_rows:
            if label not in labels_amount_dict:
                labels_amount_dict[label] = 1
            else:
                labels_amount_dict[label] = labels_amount_dict[label] + 1
    max_size = 0
    for label in labels_amount_dict:
        if label != 0 and labels_amount_dict[label] > max_size:
            max_size = labels_amount_dict[label]
    return max_size


def save_the_biggest_cluster_size_plot(lattice_size, tree_probabilities, the_biggest_cluster_sizes):
    fig, ax = plt.subplots()
    ax.plot(tree_probabilities, the_biggest_cluster_sizes)
    ax.set_title(r'The biggest cluster sizes for $L=$' + str(lattice_size))
    ax.set_xlabel(r'Tree probabilities $p$')
    ax.set_ylabel("The biggest cluster sizes")
    ax.grid()
    file_title = 'The biggest cluster sizes plot, L=' + str(lattice_size) + '.png'
    plt.savefig(file_title)


def determine_size_of_the_biggest_cluster():
    lattice_size = 100
    tree_probabilities = np.arange(0.1, 1, 0.05)
    the_biggest_cluster_sizes = []
    for tree_prob in tree_probabilities:
        _, the_biggest_cluster_size = make_simulation_of_fire_in_forest(tree_prob, lattice_size, False, True)
        the_biggest_cluster_sizes.append(the_biggest_cluster_size)

    save_the_biggest_cluster_size_plot(lattice_size, tree_probabilities, the_biggest_cluster_sizes)


if __name__ == "__main__":
    perform_some_forest_fire_simulations()
    # determine_percolation_threshold_for_various_lattice_sizes()
    # determine_size_of_the_biggest_cluster()
