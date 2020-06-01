# This Python file uses the following encoding: utf-8
from setuptools import setup, find_packages

setup(  name='timspy',
        packages=find_packages(),
        version='0.0.3',
        description='TimsPy: access for raw timsTOF Pro data for data scientists',
        long_description='TimsPy facilitates access to the raw data gathered by timsTOF Pro mass spectrometer. Directly see data in the long format in the familiar numpy array of Pandas DataFrame object.',
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
        install_requires=['numpy',
                          'pandas',
                          'scipy',
                          'timsdata',
                          'rmodel'],
        extras_require={
            'plots': ('matplotlib >= 3.0.0'),
        },
)
