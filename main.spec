# -*- mode: python ; coding: utf-8 -*-

import sys
import glob
import pathlib
sys.setrecursionlimit(10000)

block_cipher = None

# Include all necessary project files
src_folder = pathlib.Path('c:/Users/BardetJ/Documents/aoslo_pipeline/src')
added_files = [
    # Include app.py
    ('src/input_GUI/app.py', 'src/input_GUI'),
    ('src/input_GUI/config_reader.py', 'src/input_GUI'),
    
    # Include the AST module
    ('src/AST', 'src/AST'),

    # Include config file - adjust the path if needed
    ('src/configs/config.txt', 'src/configs'),
    
    # Include any config directory and all its contents
    ('src/configs', 'src/configs'),

    # Include the files directory with all its contents
    ('src/input_GUI/files', 'src/input_GUI/files'),
    
    # Include any other modules from src
]

# Add all Python files from src directory recursively
for py_file in src_folder.glob('**/*.py'):
    rel_path = py_file.relative_to(src_folder.parent)
    dest_dir = str(rel_path.parent)
    added_files.append((str(py_file), dest_dir))
for config_file in src_folder.glob('**/config*.txt'):
    rel_path = config_file.relative_to(src_folder.parent)
    dest_dir = str(rel_path.parent)
    added_files.append((str(config_file), dest_dir))
# Add image files
for ext in ['*.png', '*.PNG', '*.jpg', '*.jpeg', '*.gif', '*.bmp']:
    for img_file in src_folder.glob(f'**/files/{ext}'):
        rel_path = img_file.relative_to(src_folder.parent)
        dest_dir = str(rel_path.parent)
        added_files.append((str(img_file), dest_dir))

a = Analysis(
    ['src/input_GUI/main.py'],
    pathex=['c:\\Users\\BardetJ\\Documents\\aoslo_pipeline'],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        # Tkinter modules
        'tkinter', 'tkinter.filedialog', 'tkinter.messagebox', 'tkinter.ttk',
        'tkinter.scrolledtext', 'tkinter.font', 'tkinter.dialog', 'tkinter.colorchooser',
        'tkinter.commondialog', 'tkinter.constants', 'tkinter.dnd',
        # Project modules
        'src', 'src.AST', 'src.input_GUI', 'config_reader',
        # Other common imports
        'cv2', 'numpy', 'PIL', 'tifffile',
        # scikit-image specific imports
        'skimage', 'skimage.io', 'skimage.morphology', 'skimage.feature',
        'skimage.transform', 'skimage.util', 'skimage.filters', 'skimage.measure',
        'skimage.color', 'skimage.exposure', 'skimage.segmentation',
        # Stats modules
        'statsmodels', 'statsmodels.api', 'statsmodels.formula.api',
        'statsmodels.nonparametric', 'statsmodels.nonparametric.smoothers_lowess',
        'statsmodels.tsa', 'statsmodels.graphics',
        'patsy', 'scipy.stats',
        # Screen information module
        'screeninfo', 'fpdf',
        # Geometry modules
        'shapely', 'shapely.affinity', 'shapely.geometry',
        'rtree', 'rtree.index',
        # Numba modules
        'numba', 'numba.core', 'numba.core.types', 'numba.core.types.old_scalars',
        'numba.np', 'numba.np.ufunc', 'numba.core.typing', 'numba.core.imputils',
        'numba.core.dispatcher', 'numba.core.registry', 'numba.misc',
        'numba.misc.special', 'numba.typed', 'numba.cpython', 'numba.core.datamodel.old_models',
        'numba.cpython.old_builtins', 'numba.core.typing.old_builtins', 'numba.cpython.old_hashing',
        'numba.cpython.old_numbers', 'numba.cpython.old_tupleobj', 'numba.np.old_arraymath',
        'numba.misc.appdirs', 'numba.misc.special', 'numba.np.random', 'numba.typed.typeddict',
        'numba.typed.typedlist', 'numba.experimental.jitclass', 'numba.core.types.containers',
        'numba.core.types.npytypes', 'numba.np.random.old_distributions', 'numba.np.random.old_random_methods',
        'numba.cpython.old_mathimpl', 'numba.core.typing.old_cmathdecl', 'numba.core.typing.old_mathdecl',
        'numba.core.old_boxing',
        # Visualization libraries
        'matplotlib', 'seaborn',
        # Utilities
        'deprecated',
    ],
    hookspath=['./hooks'],
    hooksconfig={},
    runtime_hooks=['hooks/runtime-hook.py'],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='aoslo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='aoslo',
)