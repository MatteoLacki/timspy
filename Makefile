# This Makefile is for the convenience of the package developers,
# and is not meant for use by end-users.

reinstall: pyclean pipclean
	pip install . --verbose --no-cache 
py:
	python -m IPython
pyclean:
	rm -rf build dist timspy.egg-info
pipclean:
	pip uninstall timspy -y || true
	pip uninstall timspy -y || true
docs: clean_docs
	git branch gh-pages || true
	git checkout gh-pages
	pip install sphinx || true
	pip install recommonmark || true
	mkdir -p sphinx
	mkdir -p docs || True
	touch docs/.nojekyll
	sphinx-quickstart sphinx --sep --project OpenTIMS --author Lacki_and_Startek -v 0.0.1 --ext-autodoc --ext-githubpages --extensions sphinx.ext.napoleon --extensions recommonmark --makefile -q --no-batchfile
	sphinx-apidoc -f -o sphinx/source opentimspy
	cd sphinx && make html
	cp -r sphinx/build/html/* docs
	git checkout master
clean_docs:
	rm -rf sphinx
	rm -rf docs
pypi_test:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/* 
pypi:
	twine upload dist/*
test_get_TIC:
	python bin/get_TIC.py -h
	python bin/get_TIC.py "/home/matteo/Projects/bruker/BrukerMIDIA/MIDIA_CE10_precursor/20190912_HeLa_Bruker_TEN_MIDIA_200ng_CE10_100ms_Slot1-9_1_488.d" --plot
