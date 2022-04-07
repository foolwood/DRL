# https://github.com/chaoyuaw/pytorch-coviar

rm -rf build
python setup.py build_ext
python setup.py install --user
