# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
import copy


class Boid(object):
    def __init__(self, x_loc, y_loc, velocity_x, velocity_y, simulation):
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.simulation = simulation

    def is_neighbour(self, other_boid):
        if self is not other_boid:
            distance = self.distance_between(other_boid)
            angle_between = self.angle_between(other_boid)
            return distance < self.simulation.max_distance and np.abs(angle_between) < self.simulation.angle
        else:
            return False

    def distance_between(self, other_boid):
        x_difference = abs(self.x_loc - other_boid.x_loc)
        if x_difference > self.simulation.lattice_size / 2:
            x_difference = self.simulation.lattice_size - x_difference
        y_difference = abs(self.y_loc - other_boid.y_loc)
        if y_difference > self.simulation.lattice_size / 2:
            y_difference = self.simulation.lattice_size - y_difference
        return np.sqrt(x_difference * x_difference + y_difference * y_difference)

    def angle_between(self, other_boid):
        delta_x = self.delta_x(other_boid)
        delta_y = self.delta_y(other_boid)
        delta = np.array([delta_x, delta_y])
        velocity = np.array([self.velocity_x, self.velocity_y])

        scalar_product = np.dot(delta, velocity)
        delta_norm = np.linalg.norm(delta)
        velocity_norm = np.linalg.norm(velocity)
        cos = scalar_product / (delta_norm * velocity_norm)
        return np.arccos(cos)

    def delta_x(self, other_boid):
        x_difference = other_boid.x_loc - self.x_loc
        if x_difference > self.simulation.lattice_size / 2:
            x_difference = x_difference - self.simulation.lattice_size
        elif x_difference < -self.simulation.lattice_size / 2:
            x_difference = self.simulation.lattice_size - x_difference
        return x_difference

    def delta_y(self, other_boid):
        y_difference = other_boid.y_loc - self.y_loc
        if y_difference > self.simulation.lattice_size / 2:
            y_difference = y_difference - self.simulation.lattice_size
        elif y_difference < -self.simulation.lattice_size / 2:
            y_difference = self.simulation.lattice_size - y_difference
        return y_difference


class FlockSimulation:

    # constructor for simulation
    def __init__(self, boids_number, min_speed, max_speed, angle, max_distance, min_distance, weight_sep,
                 weight_allignment, weight_cohesion, time, obstacle, lattice_size=100):

        self.lattice_size = lattice_size
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.angle = angle
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.weight_sep = weight_sep
        self.weight_alignment = weight_allignment
        self.weight_cohesion = weight_cohesion
        self.time = time
        self.obstacle = obstacle

        # init of boids lists
        self.boids = []

        if self.obstacle:
            # init obstacle in random place which is a square whose length is 10% of lattice size
            self.obstacle_x1 = 0.9 * self.lattice_size * np.random.rand()
            self.obstacle_x2 = self.obstacle_x1 + 0.1 * self.lattice_size
            self.obstacle_y1 = 0.9 * self.lattice_size * np.random.rand()
            self.obstacle_y2 = self.obstacle_y1 + 0.1 * self.lattice_size

        # Boids location in each step. Used while animating
        self.locations_in_each_step = []
        self.init_boids(boids_number)

    def generate_starting_locations(self, boids_number):
        return zip(self.lattice_size * np.random.rand(boids_number),
                   self.lattice_size * np.random.rand(boids_number))

    def init_boids(self, boids_number):
        locations = self.generate_starting_locations(boids_number)
        # creating boids
        for x, y in locations:

            # in the case when location is inside the obstacle we random it once again till it is ok
            while self.location_inside_obstacle(x, y):
                x = self.lattice_size * np.random.rand()
                y = self.lattice_size * np.random.rand()

            # init some velocity values
            v_x = self.min_speed + (self.max_speed - self.min_speed) * np.random.rand()
            v_y = self.min_speed + (self.max_speed - self.min_speed) * np.random.rand()
            speed = np.linalg.norm([v_x, v_y])

            # when speed is greteat than maximum possible we decrease it to the maximum one
            if speed > self.max_speed:
                v_x = v_x / speed * self.max_speed
                v_y = v_y / speed * self.max_speed

            new_boid = Boid(x, y, v_x, v_y, self)
            self.boids.append(new_boid)

    def location_inside_obstacle(self, x, y):
        if self.obstacle:
            return self.obstacle_x1 <= x <= self.obstacle_x2 and self.obstacle_y1 <= y <= self.obstacle_y2
        else:
            return False

    # method which perform simulation
    def run_simulation(self, animate_simulation=False):
        self.memorize_current_locations_of_boids()
        for i in range(self.time):
            self.perform_step()
            print('Step ' + str(i) + ' done')

        if animate_simulation:
            self.animate_simulation()

    def perform_step(self):
        # List of new boids. We will add to it updated boids.
        new_boids = []
        for current_boid in self.boids:
            self.handle_movement_of_boid(current_boid, new_boids)

        # We replace all boids with the new ones
        self.boids = new_boids
        self.memorize_current_locations_of_boids()

    def handle_movement_of_boid(self, current_boid, new_boids):
        new_current_boid = copy.copy(current_boid)
        neighbours = self.find_neighbours_of_boid(current_boid)
        self.determine_velocity_of_boid(neighbours, new_current_boid)
        self.move_boid(new_current_boid)

        # We add to list of updated boids the current boid.
        new_boids.append(new_current_boid)

    def memorize_current_locations_of_boids(self):
        x_locations = [boid.x_loc for boid in self.boids]
        y_locations = [boid.y_loc for boid in self.boids]
        self.locations_in_each_step.append((x_locations, y_locations))

    def find_neighbours_of_boid(self, current_boid):
        neighbours = []
        for other_boid in self.boids:
            if current_boid.is_neighbour(other_boid):
                neighbours.append(other_boid)
        return neighbours

    def determine_velocity_of_boid(self, neighbours, new_current_boid):
        # when neighbours is not an empty list
        if neighbours:
            self.alignment(neighbours, new_current_boid)
            self.cohesion(neighbours, new_current_boid)
            self.separation(neighbours, new_current_boid)
            self.adjust_velocity(new_current_boid)

        # We check if current veocity will not cause that a boid hit the obstacle.
        # If it does we try to change the velocity. We do it until we are sure that
        # the boid will not hit into the obstacle
        while self.will_hit_obstacle(new_current_boid):
            self.try_avoid_obstacle(new_current_boid)

    def move_boid(self, new_current_boid):
        # moving boid by its velocity
        new_current_boid.x_loc = (new_current_boid.x_loc + new_current_boid.velocity_x) % self.lattice_size
        new_current_boid.y_loc = (new_current_boid.y_loc + new_current_boid.velocity_y) % self.lattice_size

    def alignment(self, neighbours, new_current_boid):
        neigbours_av_velocity_x = np.mean([neighbour.velocity_x for neighbour in neighbours])
        neigbours_av_velocity_y = np.mean([neighbour.velocity_y for neighbour in neighbours])

        new_current_boid.velocity_x = new_current_boid.velocity_x + self.weight_alignment * (
                neigbours_av_velocity_x - new_current_boid.velocity_x)
        new_current_boid.velocity_y = new_current_boid.velocity_y + self.weight_alignment * (
                neigbours_av_velocity_y - new_current_boid.velocity_y)

    def cohesion(self, neighbours, new_current_boid):
        av_distance = np.mean([new_current_boid.distance_between(neighbour) for neighbour in neighbours])
        for neighbour in neighbours:
            distance = new_current_boid.distance_between(neighbour)

            delta_x = new_current_boid.delta_x(neighbour)
            delta_y = new_current_boid.delta_y(neighbour)

            new_current_boid.velocity_x = new_current_boid.velocity_x + self.weight_cohesion * delta_x * (
                    distance - av_distance) / distance
            new_current_boid.velocity_y = new_current_boid.velocity_y + self.weight_cohesion * delta_y * (
                    distance - av_distance) / distance

    def separation(self, neighbours, new_current_boid):
        for neighbour in neighbours:
            distance = new_current_boid.distance_between(neighbour)

            delta_x = new_current_boid.delta_x(neighbour)
            delta_y = new_current_boid.delta_y(neighbour)

            new_current_boid.velocity_x = new_current_boid.velocity_x - self.weight_sep * (
                    delta_x * self.min_distance / distance - delta_x)
            new_current_boid.velocity_y = new_current_boid.velocity_y - self.weight_sep * (
                    delta_y * self.min_distance / distance - delta_y)

    def adjust_velocity(self, new_current_boid):
        speed = np.linalg.norm([new_current_boid.velocity_x, new_current_boid.velocity_y])
        # If speed above maximum possible we decrease it to that maximum value,
        # if below minimum we  increase it to minimum values
        if speed > self.max_speed:
            new_current_boid.velocity_x = new_current_boid.velocity_x / speed * self.max_speed
            new_current_boid.velocity_y = new_current_boid.velocity_y / speed * self.max_speed
        elif speed < self.min_speed:
            new_current_boid.velocity_x = new_current_boid.velocity_x / speed * self.min_speed
            new_current_boid.velocity_y = new_current_boid.velocity_y / speed * self.min_speed

    def will_hit_obstacle(self, new_current_boid):
        if self.obstacle:
            # we chceck gradually if a boids moving with given velocity will not hit the obstacle
            for alpha in np.linspace(0, 1, 20):
                new_x = (new_current_boid.x_loc + alpha * new_current_boid.velocity_x) % self.lattice_size
                new_y = (new_current_boid.y_loc + alpha * new_current_boid.velocity_y) % self.lattice_size

                if self.location_inside_obstacle(new_x, new_y):
                    return True
            return False
        else:
            return False

    def try_avoid_obstacle(self, new_current_boid):
        if self.obstacle:
            for alpha in np.linspace(0, 1, 20):

                # possible coordinate of a boid while moving
                new_x = (new_current_boid.x_loc + alpha * new_current_boid.velocity_x) % self.lattice_size
                new_y = (new_current_boid.y_loc + alpha * new_current_boid.velocity_y) % self.lattice_size

                if self.location_inside_obstacle(new_x, new_y):

                    # Creating array of angles by which we will try to rotate velocity vector
                    # It will be [pi/12, -pi/12, pi/11 -pi/11 ... pi]
                    angles = np.array([angle * np.array([1, -1]) for angle in [np.pi / i for i in range(12, 1, -1)]])
                    angles = angles.flatten()
                    i = 0

                    # Variables which will be values of velocity after rotations
                    new_vel_x = new_current_boid.velocity_x
                    new_vel_y = new_current_boid.velocity_y

                    # rotating velocity until possible coordinate of boid are not inside the obstacle
                    while self.location_inside_obstacle(new_x, new_y) and i < len(angles):
                        new_vel_x, new_vel_y = self.rotate_vector(new_current_boid.velocity_x,
                                                                  new_current_boid.velocity_y,
                                                                  angles[i])

                        # new possible coordinate of a boid while moving
                        new_x = (new_current_boid.x_loc + alpha * new_vel_x) % self.lattice_size
                        new_y = (new_current_boid.y_loc + alpha * new_vel_y) % self.lattice_size
                        i = i + 1

                    new_current_boid.velocity_x = new_vel_x
                    new_current_boid.velocity_y = new_vel_y
                    return

    @staticmethod
    def rotate_vector(x, y, angle):
        new_x = x * np.cos(angle) - y * np.sin(angle)
        new_y = y * np.cos(angle) + x * np.sin(angle)
        return new_x, new_y

    def draw_lattice(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.lattice_size)
        ax.set_ylim(0, self.lattice_size)
        ax.set_aspect('equal')
        for boid in self.boids:
            plt.plot(boid.x_loc, boid.y_loc, 'b.')

        rectangle = Rectangle((self.obstacle_x1, self.obstacle_y1), 0.1 * self.lattice_size, 0.1 * self.lattice_size,
                              facecolor='grey', alpha=1, edgecolor='black')
        ax.add_patch(rectangle)
        plt.show()

    def animate_simulation(self):

        fig, ax = plt.subplots()
        points, = ax.plot([], [], '.', c='#ff9900')
        frame_amount = self.time + 1
        interval_time = 50

        # Function preparing an animation
        def init_animation():
            # Setting axis parameters and title
            ax.set_xlim(0, self.lattice_size)
            ax.set_ylim(0, self.lattice_size)
            ax.set_aspect('equal')
            rectangles = []
            if self.obstacle:
                obstacle = Rectangle((self.obstacle_x1, self.obstacle_y1), 0.1 * self.lattice_size,
                                     0.1 * self.lattice_size,
                                     facecolor='#ffff66', alpha=1, edgecolor='#cc9900')
                ax.add_patch(obstacle)
                rectangles.append(obstacle)
            background = Rectangle((0, 0), self.lattice_size, self.lattice_size,
                                   facecolor='#0099ff', alpha=0.2)
            ax.add_patch(background)
            rectangles.append(background)
            points.set_data([], [])
            return points, obstacle, background

        def animate(frame):
            x_locaction, y_location = self.locations_in_each_step[frame]
            xdata = []
            ydata = []
            for x, y in zip(x_locaction, y_location):
                xdata.append(x)
                ydata.append(y)

            points.set_data(xdata, ydata)
            # rectangle = Rectangle((0, 0), self.lattice_size, self.lattice_size,
            #                      facecolor='#0099ff', alpha=0.6)
            # ax.add_patch(rectangle)
            ax.set_title("Step no: " + str(frame))
            print("Frame " + str(frame) + " animated")
            return points,  # rectangle

        anim = animation.FuncAnimation(fig, animate,
                                       init_func=init_animation,
                                       frames=frame_amount,
                                       interval=interval_time,
                                       blit=True,
                                       repeat=False)

        gif_title = "test_flocks" + \
                    " boids_no=" + str(len(self.boids)) + \
                    " L=" + str(self.lattice_size) + \
                    " T=" + str(self.time) + \
                    " v_min=" + str(self.min_speed) + \
                    " v_min=" + str(self.max_speed) + \
                    " d_min=" + str(self.min_distance) + \
                    " d_min=" + str(self.max_distance) + \
                    " w_a=" + str(self.weight_alignment) + \
                    " w_c=" + str(self.weight_cohesion) + \
                    " w_s=" + str(self.weight_sep) + \
                    " obstacle=" + str(self.obstacle)

        plt.show()
        # anim.save(gif_title + '.gif', writer='imagemagick')
        print(gif_title)


if __name__ == "__main__":
    flock_simulation = FlockSimulation(boids_number=25, min_speed=0.2, max_speed=1.8, angle=2 / 3 * np.pi,
                                       max_distance=8, min_distance=4, weight_allignment=0.2, weight_cohesion=0.1,
                                       weight_sep=0.15, lattice_size=100, time=200, obstacle=True)

    flock_simulation.run_simulation(animate_simulation=True)
