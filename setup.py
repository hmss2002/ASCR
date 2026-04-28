from setuptools import find_packages, setup

setup(
    name="ascr",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "ascr-stage1=ascr.cli.run_stage1:main",
            "ascr-train-selector=ascr.training.train_selector:main",
        ]
    },
)
