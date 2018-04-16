from . import buildin_sae_file

class EnergyShifter:

    def __init__(self, self_energy_file=buildin_sae_file):
        # load self energies
        self.self_energies = {}
        with open(self_energy_file) as f:
            for i in f:
                try:
                    line = [x.strip() for x in i.split('=')]
                    name = line[0].split(',')[0].strip()
                    value = float(line[1])
                    self.self_energies[name] = value
                except:
                    pass  # ignore unrecognizable line

    def __call__(self, energies, species):
        s = 0
        for i in species:
            s += self.self_energies[i]
        return energies - s
