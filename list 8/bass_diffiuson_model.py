import numpy as np
import matplotlib.pyplot as plt


class BassDiffusionModel(object):
    def __init__(self, M, p, q):
        self.M = M
        self.p = p
        self.q = q
        self.agents_adoption = np.array(M * [False])
        self.agents_adopted_at_each_step = []
        self.agents_adopted_by_innovation_at_each_step = []
        self.agents_adopted_by_imitation_at_each_step = []
        self.current_agent_to_adopt = 0

    def any_agents_not_adopted(self):
        return not np.all(self.agents_adoption)

    def how_many_adopted(self):
        return np.sum(self.agents_adoption == True)

    def run(self):
        while self.any_agents_not_adopted():
            A = self.how_many_adopted()
            adopted_by_innovation = int(np.ceil((self.M - A) * self.p))
            adopted_by_imitation = int(np.ceil((A - A * A / self.M) * self.q))
            adopted = adopted_by_innovation + adopted_by_imitation

            for i in range(self.current_agent_to_adopt, self.current_agent_to_adopt + adopted):
                if i < len(self.agents_adoption):
                    self.agents_adoption[i] = True

            self.current_agent_to_adopt += adopted

            self.agents_adopted_at_each_step.append(adopted)
            self.agents_adopted_by_innovation_at_each_step.append(adopted_by_innovation)
            self.agents_adopted_by_imitation_at_each_step.append(adopted_by_imitation)

    def draw_plot(self):
        plt.subplot(2, 1, 1)
        plt.plot(np.cumsum(self.agents_adopted_at_each_step), '.-')
        plt.grid()
        plt.title('Adopted')
        plt.subplot(2, 1, 2)
        plt.plot(self.agents_adopted_at_each_step, '.-', ms=10, label='adopted')
        plt.plot(self.agents_adopted_by_imitation_at_each_step, '.-', lw=0.75, label='by imitation')
        plt.plot(self.agents_adopted_by_innovation_at_each_step, '.-', lw=0.75, label='by innovation')
        plt.title('Adopted at each step')
        plt.grid()
        plt.legend(loc='best')
        plt.suptitle(r'Agentized Bass diffusion model, $p=$' + str(p) + r' $q=$' + str(q) + r' $M=$' + str(M))
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


if __name__ == "__main__":
    M = 1000
    p = 0.01
    q = 0.5

    bass_diffusion = BassDiffusionModel(M, p, q)
    bass_diffusion.run()
    bass_diffusion.draw_plot()
