# Object Grasp Annotation
Grasp annotation for single object.

## Installation
1. Install packages from Pip.
```bash
    pip install -r requirements.txt
```

2. Install dex-net.
```bash
    cd dex-net
    python setup.py develop
```

3. Install meshpy.
```bash
    cd meshpy
    python setup.py develop
```

4. Install SDFGen.
```bash
    cd SDFGen
    mkdir build && cd build
    cmake ..
    make
```

## Usage
1. Create ``models`` folder (or make a soft link) and move object models to ``models``. For each model, ``nontextured.ply`` and ``textured.obj`` are required.

2. The sampling parameters are saved in [``utils/obj_params.py``](utils/obj_params.py), where ``sample_voxel_size`` is used to sample grasp points, ``model_voxel_size`` is used to sample the model for collision detection and ``model_num_sample`` is the maximum sampling number of the model. __Do not forget to add sampling configurations before generating grasps for new models__.

3. Generate sdf files for models. You can modify ``obj_names`` in [``sdf_gen.py``](sdf_gen.py) as you need.
```bash
    python sdf_gen.py
```

4. Generate grasp annotations. You can modify ``obj_list`` as you need. Default worker number is 50 and modify it according to your CPUs. The following command will generate grasp labels for each object in .npz format, which contains ``points`` (grasp points), ``offsets`` (gripper inplane rotation angles, depths and widths), ``collision`` (collision mask, True indicates collision) and ``scores`` (grasp quality denoted by friction coefficient, lower score indicates higher quality).
```bash
    python data_gen.py
```

## Citation
Please cite these papers in your publications if it helps your research:
```
@article{fang2023robust,
  title={Robust grasping across diverse sensor qualities: The GraspNet-1Billion dataset},
  author={Fang, Hao-Shu and Gou, Minghao and Wang, Chenxi and Lu, Cewu},
  journal={The International Journal of Robotics Research},
  year={2023},
  publisher={SAGE Publications Sage UK: London, England}
}

@inproceedings{fang2020graspnet,
  title={GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping},
  author={Fang, Hao-Shu and Wang, Chenxi and Gou, Minghao and Lu, Cewu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition(CVPR)},
  pages={11444--11453},
  year={2020}
}
```
