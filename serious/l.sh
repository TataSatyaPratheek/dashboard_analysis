conda activate rdash

# 1. Uninstall EVERYTHING that depends on or is numpy-like.
# The order can sometimes matter due to how uninstallers handle dependencies.
# Start with higher-level packages. -y confirms uninstallation.
echo "Uninstalling potentially problematic packages..."
pip uninstall sentence-transformers -y
pip uninstall transformers -y
pip uninstall torch torchaision torchaudio -y # If installed via pip
pip uninstall faiss-cpu -y # Just in case, though you said it wasn't found
pip uninstall python-igraph -y
pip uninstall scikit-learn -y
pip uninstall scipy -y
pip uninstall numpy -y

# If installed via conda, also try to remove them via conda
conda uninstall sentence-transformers transformers pytorch torchvision torchaudio faiss-cpu python-igraph scikit-learn scipy numpy --force -y

# 2. Verify NumPy is GONE from the environment
python -c "import numpy; print(numpy.__version__)"
# This SHOULD now fail with ModuleNotFoundError. If it doesn't, NumPy is still somewhere.

# 3. Reinstall using Conda (preferable for M1 for core scientific packages)
echo "Reinstalling core packages via Conda..."
conda install numpy scipy scikit-learn -c conda-forge -y # Using conda-forge for robust builds
conda install pytorch torchvision torchaudio -c pytorch -y # Get PyTorch from its official channel
conda install python-igraph -c conda-forge -y
conda install faiss-cpu -c pytorch -y # Get FAISS from PyTorch channel (good M1 support)

# 4. Reinstall Python-specific packages via Pip
echo "Reinstalling pip packages..."
pip install transformers sentence-transformers

# 5. Check NumPy installation again
echo "Verifying NumPy installation..."
python -c "import numpy; print(f'NumPy version: {numpy.__version__}'); print(f'NumPy location: {numpy.__file__}'); print(hasattr(numpy, 'ndarray'))"
# This should now print the version, location (in your rdash site-packages), and True.
