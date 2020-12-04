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
	sphinx-quickstart sphinx --sep --project TimsPy --author Lacki_and_Startek -v 0.0.1 --ext-autodoc --ext-githubpages --extensions sphinx.ext.napoleon --extensions recommonmark --makefile -q --no-batchfile
	sphinx-apidoc -f -o sphinx/source timspy
	cd sphinx && make html
	cp -r sphinx/build/html/* docs
clean_docs:
	rm -rf sphinx
	rm -rf docs
pypi_test:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/* 
pypi:
	twine upload dist/*
