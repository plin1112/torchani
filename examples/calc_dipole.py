
def calculate_dipole_moment(mol):
    """
    Calculates dipole moment of the molecule.

    mol: class with atomic coordinates and point charges
    """
    from math import sqrt

    neg_charge = 0.0
    pos_charge = 0.0
    neg_center = [0.0, 0.0, 0.0]
    pos_center = [0.0, 0.0, 0.0]
    for atom in mol.atoms:
        charge = atom.charge
        x, y, z = atom.coords
        if charge < 0.0:
            neg_center[0] -= charge * x
            neg_center[1] -= charge * y
            neg_center[2] -= charge * z
            neg_charge -= charge
        elif charge > 0.0:
            pos_center[0] += charge * x
            pos_center[1] += charge * y
            pos_center[2] += charge * z
            pos_charge += charge

    if abs(pos_charge - neg_charge) > 0.001:
        print('Warning: sum of positive charges != sum of negative charges')

    dipole = 0.0
    for i in range(3):
        dipole += (neg_center[i] - pos_center[i]) * (neg_center[i] - pos_center[i])

    dipole = sqrt(dipole)
    # Convert to Debye, assume x, y, z in Angstroms
    toDebye = 4.80320425
    dipole *= toDebye * dipole

    dipole_dir = [0.0, 0.0, 0.0]
    for i in range(0, 3):
        dipole_dir[i] =  pos_center[i] / pos_charge - neg_center[i] / neg_charge

    dipole_center = [0.0, 0.0, 0.0]
    for i in range(3):
        dipole_center[i] = 0.5 * (pos_center[i] / pos_charge + neg_center[i] / neg_charge)

    return dipole, dipole_dir, dipole_center
