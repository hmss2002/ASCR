from setuptools import find_packages, setup

# Compatibility shim for tooling that still reads setup.py directly.
# pyproject.toml is the canonical package metadata source.
setup(
    name="ascr",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "ascr-stage1=ascr.cli.run_stage1:main",
            "ascr-stage1-direct=ascr.cli.run_stage1_direct:main",
            "ascr-stage1-mmada-self=ascr.cli.run_stage1_mmada_self:main",
            "ascr-stage1-mmada-self-coarse=ascr.cli.run_stage1_mmada_self_coarse:main",
            "ascr-compare-stage1-variants=ascr.cli.compare_stage1_variants:main",
            # Legacy preserved Show-o comparison entrypoint; not the project mainline.
            "ascr-compare-showo=ascr.cli.compare_showo_ascr:main",
        ]
    },
)
