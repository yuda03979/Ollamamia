from setuptools import setup, find_packages

setup(
    name='ollamamia',
    version='0.0.2',
    package_dir={"ollamamia": "src"},
    packages=["ollamamia"],
    py_modules=["agent_store", "models_manager", "ollamamia"],
    install_requires=[
        "numpy~=1.26.0",
        "scikit-learn~=1.4.2",
        "pydantic~=2.9.1",
        "setuptools~=70.0.0",
        "ollama~=0.4.4"
    ],
)