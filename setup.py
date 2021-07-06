from setuptools import setup

requirements = [
    "pyyaml",
    "tensorflow",
    "tensorboard",
]

setup(
    name="fasterrcnn",
    version="0.1.0",
    py_modules=["main"],
    install_requires=requirements,
    entry_points="""
        [console_scripts]
        fasterrcnn=main:main
    """,
)
