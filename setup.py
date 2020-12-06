# This Python file uses the following encoding: utf-8
from setuptools import setup, find_packages

setup(  name='timspy',
        packages=find_packages(),
        version='0.9.1',
        description='TimsPy: access for raw timsTOF Pro data for data scientists',
        long_description='TimsPy facilitates access to the raw data gathered by timsTOF Pro mass spectrometer. Directly see data in the long format in the familiar Pandas DataFrame object.',
        author='MatteoLacki',
        author_email='matteo.lacki@gmail.com',
        url='https://github.com/MatteoLacki/timspy.git',
        keywords=['timsTOFpro', 'numpy', 'data science','mass spectrometry'],
        classifiers=['Development Status :: 1 - Planning',
                     'License :: OSI Approved :: BSD License',
                     'Intended Audience :: Science/Research',
                     'Topic :: Scientific/Engineering :: Chemistry',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8'],
        install_requires=['pandas',
                          'opentimspy',
                          'matplotlib',
                          'opentims_bruker_bridge',
                          'tqdm'],
        extras_require={
            'vaex': ['vaex-core',
                     'vaex-hdf5',
                     'h5py'],
        },
        scripts = [
            'bin/tims2hdf5.py'
        ]
)
