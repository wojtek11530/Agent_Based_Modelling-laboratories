import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import copy
from matplotlib import animation


class TrafficModel(object):
    # class agent with parameters: kind and localization
    class Car:
        def __init__(self, location, traffic_model):
            self.traffic_model = traffic_model
            self.velocity = np.random.randint(self.traffic_model.max_velocity)
            self.location = location

    def __init__(self, road_length, max_velocity, car_density, prob_of_slowing, time):
        self.road_length = road_length
        self.road_occupation = self.road_length * [False]
        self.max_velocity = max_velocity
        self.prob_of_slowing = prob_of_slowing
        self.time = time

        self.car_density = car_density
        car_numbers = int(self.road_length * car_density)

        self.cars = []
        self.init_cars(car_numbers)

        self.road_occupation_in_each_step = []

    def init_cars(self, car_numbers):
        road_length = len(self.road_occupation)
        locations = np.random.choice(np.arange(road_length), car_numbers, replace=False)
        for location in locations:
            new_car = self.Car(location, self)
            self.road_occupation[location] = True
            self.cars.append(new_car)

    def run_simulation(self):
        # self.draw_road()

        self.road_occupation_in_each_step.append(self.road_occupation)

        for i in range(self.time):
            self.accelerate_cars()
            self.slow_down_cars()
            self.random_slow_down_cars()
            self.move_cars()

            self.road_occupation_in_each_step.append(copy.copy(self.road_occupation))
            print("Step " + str(i))

        self.animate_simulation()

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


if __name__ == "__main__":
    myTraffic = TrafficModel(road_length=30,
                             max_velocity=5,
                             prob_of_slowing=0.6,
                             car_density=0.3, time=30)
    myTraffic.run_simulation()
