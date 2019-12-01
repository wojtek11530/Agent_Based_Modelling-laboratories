
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation


class GameOfLife:
    DEAD = 0
    ALIVE = 1

    def __init__(self, text_file_source, end_time):
        self.lattices_in_each_step = []
        self.end_time = end_time
        self.lattice = self.init_lattice_from_file(text_file_source)

    def init_lattice_from_file(self, text_file_source):
        lattice = []

        # the file should contain of equal rows of 0s and 1s, 1 means ALIVE, 0 means DEAD
        file = open(text_file_source)
        lines_in_file = file.readlines()
        for line in lines_in_file:
            row = []
            for char in line:
                if char == '1':
                    row.append(self.ALIVE)
                elif char == '0':
                    row.append(self.DEAD)
            lattice.append(row)
        return lattice

    def run_simulation(self):
        current_lattice = np.copy(self.lattice)
        self.lattices_in_each_step.append(current_lattice)
        current_time = 0

        while current_time < self.end_time:
            new_lattice = np.copy(self.lattice)
            for i in range(0, len(self.lattice)):
                for j in range(0, len(self.lattice[0])):
                    living_neighbours = self.count_living_neighbours(i, j)

                    # the case when the cell is ALIVE
                    if self.lattice[i][j] == self.ALIVE:

                        # if the number of ALIVE neighbours is less than 2 or greater than 3
                        # then the cell is DEAD
                        if living_neighbours < 2 or living_neighbours > 3:
                            new_lattice[i][j] = self.DEAD
                        # otherwise it stays ALIVE
                        else:
                            new_lattice[i][j] = self.ALIVE
                    # else, which means the cell is DEAD
                    else:
                        # if the number of ALIVE neighbours is 3 then the cell is ALIVE
                        if living_neighbours == 3:
                            new_lattice[i][j] = self.ALIVE
                        # otherwise it stays DEAD
                        else:
                            new_lattice[i][j] = self.DEAD

            self.lattices_in_each_step.append(new_lattice)
            self.lattice = new_lattice
            current_time = current_time+1

    # count amount of
    def count_living_neighbours(self, i, j):

        row_size = len(self.lattice)
        column_size = len(self.lattice[0])

        # at the beginning there is zero living neighbours counted
        living_neighbours = 0

        # We go through all neighbours and if it is alive we increase living_neighbours
        if i - 1 >= 0 and j - 1 >= 0 and self.lattice[i - 1][j - 1] == self.ALIVE:
            living_neighbours = living_neighbours + 1
        if i - 1 >= 0 and self.lattice[i - 1][j] == self.ALIVE:
            living_neighbours = living_neighbours + 1
        if i - 1 >= 0 and j + 1 < column_size and self.lattice[i - 1][j + 1] == self.ALIVE:
            living_neighbours = living_neighbours + 1
        if j - 1 >= 0 and self.lattice[i][j - 1] == self.ALIVE:
            living_neighbours = living_neighbours + 1
        if j + 1 < column_size and self.lattice[i][j + 1] == self.ALIVE:
            living_neighbours = living_neighbours + 1
        if i + 1 < row_size and j - 1 >= 0 and self.lattice[i + 1][j - 1] == self.ALIVE:
            living_neighbours = living_neighbours + 1
        if i + 1 < row_size and self.lattice[i + 1][j] == self.ALIVE:
            living_neighbours = living_neighbours + 1
        if i + 1 < row_size and j + 1 < column_size and self.lattice[i + 1][j + 1] == self.ALIVE:
            living_neighbours = living_neighbours + 1

        return living_neighbours

    def animate_simulation(self):
        row_size = len(self.lattice)
        column_size = len(self.lattice[0])

        # Function preparing an animation
        def init_animation():

            # Setting axis parameters and title
            ax.set_xlim(0, row_size)
            ax.set_ylim(0, column_size)
            ax.set_aspect('equal')
            ax.set_axis_off()

            patches_to_return = []
            for i in range(0, row_size):
                for j in range(0, column_size):
                    rectangle = patches[i][j]
                    # Adding to the plot rectangles
                    ax.add_patch(rectangle)
                    patches_to_return.append(rectangle)
            return patches_to_return

        def animate(step):

            patches_to_return = []
            lattice_to_draw = self.lattices_in_each_step[step]  # The state of lattice at the given step
            for i in range(0, row_size):
                for j in range(0, column_size):
                    color = 'white'  # default color
                    if lattice_to_draw[i][j] == self.ALIVE:
                        color = 'red'

                    rectangle: Rectangle = patches[i][j]
                    rectangle.set_facecolor(color)  # changing color of the rectangle
                    patches_to_return.append(rectangle)

            return patches_to_return

        fig, ax = plt.subplots()
        patches = []  # list of being drawn patches (rectangles)

        for i in range(0, row_size):
            patches_in_row = []
            for j in range(0, column_size):
                # We initilize rectangles representing each cell of lattice
                rectangle = Rectangle((j, row_size - i - 1), 1, 1, edgecolor='black', alpha=1)
                patches_in_row.append(rectangle)
            patches.append(patches_in_row)

        frame_amount = len(self.lattices_in_each_step)
        interval_time = 100
        anim = animation.FuncAnimation(fig, animate,
                                       init_func=init_animation,
                                       frames=frame_amount,
                                       interval=interval_time,
                                       blit=True,
                                       repeat=False)
        # gif_title = "Fire in forest L = " + str(lattice_size) + " p = " + str(prob_of_tree)
        # anim.save(gif_title + '.gif', writer='imagemagick')
        plt.show()


if __name__ == "__main__":
    game_of_life = GameOfLife('lattice.txt', 100)
    game_of_life.run_simulation()
    game_of_life.animate_simulation()
