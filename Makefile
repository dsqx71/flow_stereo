all:
	cd utils; python cython_setup.py build_ext; mv ./build/lib.linux-x86_64-2.7/util_cython.so ./; rm -rf build; cd ../
clean:
	cd utils; rm *.so *.c; cd ../