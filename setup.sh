git submodule init
git submodule update
cd flashinfer
git submodule init
git submodule update
pip install --no-build-isolation --verbose .
cd ..
python reproduce.py --metadata-dir ./trace --mode graph