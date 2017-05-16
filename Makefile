all:
	# build Cython extension
	cd cython; python setup.py build_ext; mv ./build/lib*/util_cython.so ./; rm -rf build; cd ../;
	# python extension
	pip install lmdb
	pip install protobuf==3.0.0b2
	# pip install git+https://github.com/tody411/GuidedFilter.git --user
clean:
	# remove Cython extension
	cd cython; rm *.so *.c; cd ../;
	# remove Cuda extension
	cd operator; rm *.o *.d; cd ../;