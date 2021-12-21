try: from setuptools import setup
except: from distutils.core import setup

requirements = [line.replace('\n','') for line in open('requirements.txt','r')]
requirements = [line for line in requirements
				if line and not line.startswith('#')]

setup(	long_description="Extracts text from map images", #open("README.rst").read(), 
	name="""mapocr""",
	license="""MIT""",
	author="""Karim Bahgat""",
	author_email="""karim.bahgat.norway@gmail.com""",
	version="""0.1.0""",
	keywords="""map text character recognition""",
	packages=['mapocr'],
    install_requires=requirements,
	classifiers=['License :: OSI Approved', 'Programming Language :: Python', 'Development Status :: 4 - Beta', 'Intended Audience :: Developers', 'Intended Audience :: Science/Research', 'Intended Audience :: End Users/Desktop'],
	description="""Extracts text from map images""",
	)
