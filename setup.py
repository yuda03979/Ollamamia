from setuptools import setup, find_packages

setup(
    name='ollamamia',
    version='0.0.2',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    py_modules=["agent_store", "models_manager", "ollamamia"],
    exclude_package_data={'': ['.env']},
    install_requires=[
        'ollama==0.4.4',
        'tqdm==4.66.4',
        'readchar==4.0.5',
        'termcolor==2.5.0',
        "setuptools==69.5.1"
    ],
)
