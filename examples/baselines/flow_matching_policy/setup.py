from setuptools import setup, find_packages

setup(
    name="flow_matching_policy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "diffusers",
        "tensorboard",
        "wandb",
        "mani_skill",
        "diffusion_policy",
    ],
    description="A minimal setup for Flow Matching Policy for ManiSkill",
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
)
