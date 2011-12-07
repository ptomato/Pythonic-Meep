all:
	python setup.py build_ext --inplace

clean:
	python setup.py clean && \
	rm -f _meep.so meep.pyc
	
docs:
	cd docs && \
	make html

.PHONY: all clean docs
