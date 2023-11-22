from setuptools import find_packages, setup

setup(
    name="3dvqa",
    version="1.0",
    author="3dvqa",
    url="https://github.com/MunzerDw/guided-research.git",
    description="",
    packages=find_packages(include=("lib", "model")),
    install_requires=[
        "plyfile",
        "tqdm",
        "trimesh",
        "pytorch-lightning==1.6.5",
        "scipy",
        "open3d",
        "wandb",
        "hydra-core",
        "h5py",
    ],
)
