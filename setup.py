from setuptools import find_packages, setup

# Compatibility shim for tooling that still reads setup.py directly.
# pyproject.toml is the canonical package metadata source.
setup(
    name="ascr",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": ["numpy>=1.24", "pillow>=10.0.0", "pytest>=7"],
        "local-vlm": ["numpy>=1.24", "pillow>=10.0.0"],
        "judge": ["openai>=1.50.0", "pillow>=10.0.0"],
        "qwen-vl": [
            "torch>=2.4",
            "torchvision>=0.19",
            "transformers>=4.57.0",
            "accelerate>=1.1.0",
            "qwen-vl-utils>=0.0.14",
            "safetensors>=0.4.5",
            "sentencepiece>=0.2.0",
            "pillow>=10.0.0",
            "av>=15.0.0",
        ],
        "lumina": [
            "torch>=2.4",
            "torchvision>=0.19",
            "transformers==4.46.2",
            "diffusers>=0.30.0",
            "accelerate>=0.30.0",
            "safetensors>=0.4.5",
            "sentencepiece>=0.2.0",
            "pillow>=10.0.0",
        ],
        "mmada": [
            "torch>=2.4",
            "torchvision>=0.19",
            "transformers==4.46.2",
            "accelerate>=0.30.0",
            "safetensors>=0.4.5",
            "sentencepiece>=0.2.0",
            "pillow>=10.0.0",
        ],
        "showo": [
            "torch==2.2.1",
            "torchvision==0.17.1",
            "transformers==4.41.1",
            "diffusers==0.30.1",
            "accelerate==0.21.0",
            "omegaconf==2.3.0",
            "einops==0.6.0",
            "safetensors==0.4.3",
            "pillow==10.3.0",
            "numpy==1.24.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "ascr-stage1=ascr.cli.run_stage1:main",
            "ascr-stage1-direct=ascr.cli.run_stage1_direct:main",
            "ascr-stage1-mmada-self=ascr.cli.run_stage1_mmada_self:main",
            "ascr-stage1-mmada-self-coarse=ascr.cli.run_stage1_mmada_self_coarse:main",
            "ascr-preflight=ascr.cli.preflight:main",
            "ascr-teacher-distill=ascr.distill.teacher:main",
            "ascr-teacher-audit=ascr.distill.audit:main",
            "ascr-teacher-export=ascr.distill.export_dataset:main",
            "ascr-lumina-native-audit=ascr.cli.lumina_native_audit:main",
            "ascr-train-lumina-evaluator=ascr.training.train_lumina_evaluator:main",
            "ascr-compare-stage1-variants=ascr.cli.compare_stage1_variants:main",
            # Legacy preserved Show-o comparison entrypoint; not the project mainline.
            "ascr-compare-showo=ascr.cli.compare_showo_ascr:main",
        ]
    },
)
