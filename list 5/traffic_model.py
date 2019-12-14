import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import copy
from matplotlib import animation


class TrafficModel(object):

    class Car:
        def __init__(self, location, traffic_model):
            self.traffic_model = traffic_model

            self.velocity = 0
            self.location = location

    def __init__(self, road_length, max_velocity, car_density, prob_of_slowing, time):
        self.road_length = road_length
        self.road_occupation = self.road_length * [False]
        self.max_velocity = max_velocity
        self.prob_of_slowing = prob_of_slowing
        self.time = time

        self.car_density = car_density
        self.cars = []
        car_numbers = int(np.ceil(self.road_length * car_density))
        self.init_cars(car_numbers)

        self.average_velocity_in_each_step = np.zeros(self.time + 1)
        self.road_occupation_in_each_step = []

    def init_cars(self, car_numbers):
        road_length = len(self.road_occupation)
        locations = np.random.choice(np.arange(road_length), car_numbers, replace=False)
        for location in locations:
            new_car = self.Car(location, self)
            self.road_occupation[location] = True
            self.cars.append(new_car)

    def run_simulation(self, animate_simulation=False):

        self.road_occupation_in_each_step.append(self.road_occupation)

        self.average_velocity_in_each_step[0] = self.calculate_average_velocity()

        for i in range(self.time):
            self.accelerate_cars()
            self.slow_down_cars()
            self.random_slow_down_cars()
            self.move_cars()
            self.road_occupation_in_each_step.append(copy.copy(self.road_occupation))
            self.average_velocity_in_each_step[i] = self.calculate_average_velocity()
            print("Step " + str(i))

        if animate_simulation:
            self.animate_simulation()

    def calculate_average_velocity(self):
        velocities = []
        for car in self.cars:
            velocities.append(car.velocity)
        return np.mean(velocities)

    def get_average_velocity_for_whole_simulation(self):
        return np.mean(self.average_velocity_in_each_step)

    def draw_road(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.road_length)
        ax.set_aspect('equal')

        for loc in range(self.road_length):
            cell = self.road_occupation[loc]
            color = 'white'
            if cell is True:
                color = 'red'
            rectangle = Rectangle((loc, 0), 1, 1, facecolor=color, alpha=1, edgecolor='black')
            ax.add_patch(rectangle)

        plt.show()

    def accelerate_cars(self):
        for car in self.cars:
            if car.velocity < self.max_velocity:
                car.velocity = car.velocity + 1

    def slow_down_cars(self):
        for car in self.cars:
            for i in range(1, car.velocity):
                if self.road_occupation[(car.location + i) % self.road_length] is True:
                    car.velocity = i - 1

    def random_slow_down_cars(self):
        for car in self.cars:
            if car.velocity > 0 and np.random.rand() < self.prob_of_slowing:
                car.velocity = car.velocity - 1

    def move_cars(self):
        for car in self.cars:
            self.road_occupation[car.location] = False
            car.location = (car.location + car.velocity) % self.road_length
            self.road_occupation[car.location] = True

    def animate_simulation(self):
        # Function preparing an animation
        def init_animation():
            # Setting axis parameters and title
            ax.set_xlim(0, self.road_length)
            ax.set_ylim(-1, 2)
            ax.set_aspect('equal')
            ax.set_axis_off()
            patches_to_return = []
            return patches_to_return

        def animate(frame):
            patches_to_return = []
            road_to_draw = self.road_occupation_in_each_step[frame]

            for loc in range(self.road_length):
                cell = road_to_draw[loc]
                color = 'white'
                background = Rectangle((loc, 0), 1, 1, facecolor=color, alpha=1, edgecolor='grey')
                ax.add_patch(background)
                if cell is True:
                    color = 'red'
                    rectangle1 = Rectangle((loc + 0.05, 0.20), 0.95, 0.3, facecolor=color, alpha=1)
                    rectangle2 = Rectangle((loc + 0.25, 0.50), 0.3, 0.2, facecolor=color, alpha=1)
                    tire1 = Circle((loc + 0.25, 0.1), 0.1, color='black')
                    tire2 = Circle((loc + 0.75, 0.1), 0.1, color='black')
                    ax.add_patch(rectangle1)
                    ax.add_patch(rectangle2)
                    ax.add_patch(tire1)
                    ax.add_patch(tire2)

            ax.set_title("Step no: " + str(frame))
            print("Frame " + str(frame) + " animated")
            return patches_to_return

        fig, ax = plt.subplots()
        frame_amount = self.time + 1
        interval_time = 200
        anim = animation.FuncAnimation(fig, animate,
                                       init_func=init_animation,
                                       frames=frame_amount,
                                       interval=interval_time,
                                       blit=True,
                                       repeat=False)

        gif_title = "Traffic " + \
                    "L=" + str(self.road_length) + \
                    " p=" + str(self.prob_of_slowing) + \
                    " rho=" + str(self.car_density) + \
                    " V_max=" + str(self.max_velocity) + \
                    " time=" + str(self.time)

        anim.save(gif_title + '.gif', writer='imagemagick')
        print(gif_title)

    def visualize_animation(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.road_length)
        ax.set_ylim(0, self.time + 1)
        ax.set_aspect('equal')

        step = 0
        for road_occupation in self.road_occupation_in_each_step:
            for cell_loc in range(self.road_length):
                if road_occupation[cell_loc] is True:
                    rectangle = Rectangle((cell_loc, step), 1, 1, facecolor='black', alpha=1)
                    ax.add_patch(rectangle)

            step = step + 1

        plt.xlabel(r'Position')
        plt.ylabel(r'Step')
        plt.title(r'Visualization of simulation' +
                  r', $p=$' + str(self.prob_of_slowing) +
                  r', $\rho=$' + str(self.car_density) +
                  r', $L=$' + str(self.road_length) +
                  r', $V_{max}=$' + str(self.max_velocity) +
                  r', $T=$' + str(self.time))
        plt.grid()
        plot_title = 'Visualization of simulation ' + \
                     "L=" + str(self.road_length) + \
                     " p=" + str(self.prob_of_slowing) + \
                     " rho=" + str(self.car_density) + \
                     " V_max=" + str(self.max_velocity) + \
                     " time=" + str(self.time)
        plt.savefig(plot_title + '.png')


def task_one():
    for prob in [0.1, 0.3, 0.6, 0.9]:
        myTraffic = TrafficModel(road_length=30,
                                 max_velocity=5,
                                 prob_of_slowing=prob,
                                 car_density=0.3, time=100)
        myTraffic.run_simulation(animate_simulation=True)


def task_two():
    densities = np.arange(0.01, 0.99, 0.01)
    probabilities = [0, 0.1, 0.2, 0.3, 0.9]
    for prob in probabilities:
        N = 10
        average_velocities = np.zeros((N, len(densities)))
        for i in range(N):
            j = 0
            for density in densities:
                traffic_simulation = TrafficModel(road_length=30,
                                                  max_velocity=5,
                                                  prob_of_slowing=prob,
                                                  car_density=density, time=100)
                traffic_simulation.run_simulation(animate_simulation=False)

                average_velocities[i][j] = traffic_simulation.get_average_velocity_for_whole_simulation()
                j = j + 1

        draw_av_velocities_plot(densities, np.mean(average_velocities, axis=0), prob)

    plt.xlabel(r'density $\rho$')
    plt.ylabel(r'Average velocity')
    plt.title(r'Dependency between average velocities and cars densities')
    plt.grid()
    file_title = 'av_velocities_plot1' + '.png'
    plt.legend()
    plt.savefig(file_title)


def draw_av_velocities_plot(densities, average_velocities, slowing_probability):
    plt.plot(densities, average_velocities, '-.', label=r'p=' + str(slowing_probability))


def task_three():
    densities = [0.1, 0.2, 0.6]
    probability = 0.3
    for density in densities:
        myTraffic = TrafficModel(road_length=100,
                                 max_velocity=5,
                                 prob_of_slowing=probability,
                                 car_density=density, time=100)

        myTraffic.run_simulation(animate_simulation=False)
        myTraffic.visualize_animation()


if __name__ == "__main__":
    # task_one()
    # task_two()
    task_three()
