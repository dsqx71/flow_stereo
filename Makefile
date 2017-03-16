all:
	# build Cython extension
	cd cython; python setup.py build_ext; mv ./build/lib*/util_cython.so ./; rm -rf build; cd ../;
	# install guided filter
	# pip install git+https://github.com/tody411/GuidedFilter.git --user
clean:
	# remove Cython extension
	cd cython; rm *.so *.c; cd ../;
	# remove Cuda extension
	cd operator; rm *.o *.d; cd ../;