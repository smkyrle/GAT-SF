def multiple_pose_check(lig, pose_1=False):

    ###########################################
    # Function: Transform ligand.pdbqt        #
    # poses/models into pdbqt string blocks   #
    #                                         #
    # Inputs: ligand.pdbqt filepath           #
    #                                         #
    # Output: List of model/pose pdbqt string #
    # blocks                                  #
    ###########################################

    pdbqt_pose_blocks = list()
    lig_text = open(lig, 'r').read()
    lig_poses = lig_text.split('MODEL')
    for pose in lig_poses:
        lines = pose.split('\n')
        clean_lines = [line for line in lines if not line.strip().lstrip().isnumeric() and 'ENDMDL' not in line]
        if len(clean_lines) < 3:
            pass
        else:
            pose = '\n'.join(clean_lines)
            pdbqt_pose_blocks.append(pose)

    return pdbqt_pose_blocks

def clean_lines(lines):
    lines = [line for line in lines if not line.strip().lstrip().isnumeric() and 'ENDMDL' not in line]
    return lines


def get_poses(ligand_pdbqt, poses=[1]):
    ligand_pdbqt = ligand_pdbqt.split('MODEL')
    ligand_pdbqt = [l.split('\n') for l in ligand_pdbqt]
    ligand_pdbqt = [clean_lines(l) for l in ligand_pdbqt]
    ligand_pdbqt = ['\n'.join(l) for l in ligand_pdbqt if len(l) >= 3]
    return [ligand_pdbqt[pose-1] for pose in poses]
