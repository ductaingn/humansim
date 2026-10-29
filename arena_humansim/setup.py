import os
from glob import glob

from setuptools import find_packages, setup

package_name = "arena_humansim"


def _tree_data_files(src_root: str, share_subdir: str) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    if not os.path.isdir(src_root):
        return out
    for dirpath, _dirnames, filenames in os.walk(src_root):
        yamls = [os.path.join(dirpath, f) for f in filenames if f.endswith(".yaml")]
        if not yamls:
            continue
        rel = os.path.relpath(dirpath, src_root)
        target = f"share/{package_name}/{share_subdir}" if rel == "." else f"share/{package_name}/{share_subdir}/{rel}"
        out.append((target, yamls))
    return out


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    package_data={"arena_humansim.local_planner": ["neurosfm.npz", "NEUROSFM_LICENSE"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml") + glob("config/*.md") + glob("config/*.rviz")),
        ("share/" + package_name + "/config/agent_types", glob("config/agent_types/*.yaml")),
        ("share/" + package_name + "/config/benchmark", glob("config/benchmark/*.yaml")),
        ("share/" + package_name + "/config/scenarios", glob("config/scenarios/*.yaml")),
        *_tree_data_files("config/evaluation", "config/evaluation"),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=[
        "setuptools",
        "attrs",
        "cattrs",
        "pyyaml",
        "numpy",
        "scipy",
        "numba",
        "py_trees",
        "pyastar2d",
        "pydantic",
        "tqdm",
        "pyarrow>=14",
        "rosbags",
    ],
    extras_require={
        "test": ["pytest>=7", "pytest-xdist", "hypothesis>=6", "pandas>=2.0", "arena_humansim[socialgail,nsp,robot]"],
        "socialgail": ["torch>=2.2", "torch-geometric>=2.5"],
        "nsp": ["torch>=2.2"],
        "sarl": ["torch>=2.2"],
        "dsrnn": ["torch>=2.2"],
        "drlvo": ["torch>=2.2", "stable_baselines3>=2.0", "gymnasium>=0.29"],
        "cadrl": ["torch>=2.2"],
        "robot": ["arena_humansim[sarl,dsrnn,drlvo,cadrl]"],
    },
    zip_safe=True,
    maintainer="voshch",
    maintainer_email="dev@voshch.dev",
    description="Arena human simulator: modular, deterministic pedestrian simulation",
    license="MIT",
    entry_points={
        "console_scripts": [
            "arena_humansim_node = arena_humansim.core.agent_manager:main",
            "benchmark = arena_humansim.utils.benchmark:main",
            "arena_humansim_render = arena_humansim.utils.renderer:main",
            "evaluate = arena_humansim.utils.evaluation.cli.__main__:main",
        ],
    },
)
