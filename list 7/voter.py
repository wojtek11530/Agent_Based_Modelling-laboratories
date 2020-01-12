import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import multiprocessing as mp
import datetime

DOWN = 0
UP = 1


def simulate_q_voter_model(graph: nx.Graph, q, p, f):
    graph = graph.copy()
    prepare_simulation(graph)
    N = len(graph)
    steps_no = 50
    # in each simulation we perform steps_no of steps
    for i in range(steps_no):
        perform_step(N, f, graph, p, q)
    # after a simulation we calculate concentration
    concentration = calculate_concentration(graph)
    return concentration


def prepare_simulation(graph: nx.Graph):
    # at the beginning all agents are DOWN (non-adopted)
    nx.set_node_attributes(graph, DOWN, 'state')


def perform_step(N, f, graph, p, q):
    # in each step we perform N elementary steps, where N is a number of agents in the graph
    for j in range(N):
        elementary_step(graph, q, p, f)


def elementary_step(graph, q, p, f):
    # we randomly pick one agent
    idx = np.random.choice(len(list(graph.nodes)))
    node = list(graph.nodes)[idx]

    # we check if the agent is individual
    if np.random.random() < p:
        # if individual we change its state with probability f
        if np.random.random() < f:
            graph.nodes[node]['state'] = DOWN if graph.node[node]['state'] == UP else UP
    else:
        # agent is conformist
        # we take q its neighbours
        node_neighbours = list(graph.neighbors(node))
        q_neighbours = []
        i = 0
        while len(q_neighbours) < q:
            how_many_to_add = q - len(q_neighbours)
            if len(node_neighbours) >= how_many_to_add:
                q_neighbours.extend(np.random.choice(node_neighbours, how_many_to_add, replace=False))
            else:
                q_neighbours += node_neighbours
            # when there is less then q neighbours we wll take the next ones from the neighbours of one of
            # the agent's neighbours
            node_neighbours = list(graph.neighbors(q_neighbours[i]))
            i += 1

        # we check if the sate of all q-neighbours is the same
        state_of_first_neighbour = graph.nodes[q_neighbours[0]]['state']
        if all(graph.nodes[q_neighbour]['state'] == state_of_first_neighbour for q_neighbour in q_neighbours):
            graph.nodes[node]['state'] = state_of_first_neighbour


def calculate_concentration(graph):
    states = np.array(list(nx.get_node_attributes(graph, 'state').values()))
    # concentration as the rate of DOWN agents
    concentration = np.mean(states == DOWN)
    return concentration


if __name__ == "__main__":

    # we perform simulation for complete graph with n=100 nodes
    n = 100
    graph = nx.complete_graph(n)

    '''
    q = 4
    f = 0.5
    p = 0.2
    concentration_result = simulate_q_voter_model(graph, q, p, f)
    print(concentration_result)
    '''

    q = 4
    # for each case we make simulations_no of MC simulations
    simulations_no = 50
    f_array = [0.2, 0.3, 0.4, 0.5]
    # simulations made for different values of flexibility f
    print("Start " + str(datetime.datetime.now().time()) + '\n')
    for f in f_array:
        # array of concentrations for different p for given f
        concentrations = []
        step = 0.02
        p_array = np.arange(0, 1 + step, step)
        # for each f we perform simulations for different values of probabiliy of being individual p
        for p in p_array:
            # multiprocessing applied
            pool = mp.Pool(mp.cpu_count())
            concentrations_for_p = pool.starmap(simulate_q_voter_model,
                                                [(graph, q, p, f) for sim_no in range(simulations_no)])

            # concentration as average value of MC simulations
            concentrations.append(np.mean(concentrations_for_p))
            print('simulations done, f=' + str(f) + ', p=' + str(p))

        plt.plot(p_array, concentrations, 'o', fillstyle='none', label=r'$f=$' + str(f))
        print(str(datetime.datetime.now().time()) + ' simulations ended for f=' + str(f) + '\n')

    print(str(datetime.datetime.now().time()) + 'All simulations ended')
    plt.xlabel(r'$p$')
    plt.ylabel('c')
    plt.legend()
    plt.title(r'Concentration of adopted in the situation model in CG, $n=$' + str(n))
    plt.grid()
    file_title = 'Concentration of adopted plot, CG  ' + 'n=' + str(n) + ' sim_no=' + str(simulations_no) + '.png'
    plt.savefig(file_title)
    # plt.show()
