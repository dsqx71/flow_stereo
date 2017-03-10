all:
    # build cython extension
	cd cython; python setup.py build_ext; mv ./build/lib*/util_cython.so ./; rm -rf build; cd ../;
	# install guided filter
	# pip install git+https://github.com/tody411/GuidedFilter.git --user
clean:
    # remove cython extension
	cd cython; rm *.so *.c; cd ../