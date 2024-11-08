# -*- coding: utf-8 -*-

import glob
from setuptools import setup, find_packages

setup(
    name='chem_ant',
    version='0.1.2',
    url='https://github.com/akuroiwa/chem-ant',
    # # PyPI url
    # download_url='',
    license='GNU/GPLv3+',
    author='Akihiro Kuroiwa, ChatGPT of OpenAI, Perplexity AI',
    author_email='akuroiwa@env-reform.com, ',
    description='Select materials to output molecules similar to the target molecule with MCTS Solver and Genetic Programming.',
    # long_description="\n%s" % open('README.md').read(),
    long_description=open("README.md", "r").read(),
    long_description_content_type='text/markdown',
    zip_safe=False,
    python_requires=">=3.8, !=3.10.*",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    platforms='any',
    keywords=['evolutionary algorithms', 'genetic programming', 'gp', 'mcts', 'mcts solver', 'cheminformatics', 'chemoinformatics'],
    packages=find_packages(),
    include_package_data=True,
    install_requires=['rdkit',
                      # 'global-chem',
                      # 'global-chem-extensions',
                      # "global-chem @ git+https://github.com/Global-Chem/global-chem.git#subdirectory=global_chem",
                      # "global-chem-extensions @ git+https://github.com/Global-Chem/global-chem.git#subdirectory=global_chem_extensions",
                      'mcts',
                      'deap',
                      'mcts-solver',
                      'pandas',
                      'biopython',
                      'openbabel-wheel'],
    extras_require={
        "classification": ["transformers", "chem_classification"]},
    entry_points={
        'console_scripts': [
            'similarity-ant = chem_ant.similarity_ant:console_script',
            'similarity-mcts = chem_ant.similarity_mcts:console_script',
            'similarity-genMols = chem_ant.similarity_mcts:console_script2',
            'create-vina-config = chem_ant.create_vina_config:main',
            'prepare-experiment = chem_ant.prepare_experiment:main',
            'select-ligands = chem_ant.select_ligands:main',
            'run-experiment = chem_ant.run_experiment:main'
            ]},
    data_files=[
        ('', glob.glob('chem_ant/*.csv'))
        ],
)
