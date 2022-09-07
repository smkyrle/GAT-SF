# GAT-SF

---
Imperial College London Bioinformatics and Theoretical Systems Biology MSc thesis. The focus of this project was to develop a scoring function for prediction of receptor-ligand binding strength using graph-representation learning.

GAT-SF was trained using data from prior [publication](https://doi.org/10.1016/j.jare.2022.07.001) and [scoring function](https://github.com/SMVDGroup/SCORCH), SCORCH. Training data included docked poses of receptor-ligand complexes from [PDBbind](http://pdbbind.org.cn/), [Iridium](https://www.eyesopen.com/iridium-database), [Binding MOAD](https://bindingmoad.org/Home/download), labelled based on binding affinity and RMSD from the native crystal poses. In addition to docked actives, decoys generated using [DeepCoy](https://github.com/oxpig/DeepCoy) were also included.
