import os
from pathlib import Path
import zipfile


def zip_source(filename):
    folder = Path('.')
    files = filter(
        lambda x: ('_pycache_' not in str(x)),
        [f for d in ['codes', 'codes_two_stage', 'scripts'] for f in folder.glob(f'{d}/**/*')]
    )
    with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for entry in files:
            zip_file.write(entry, entry.relative_to(folder))


out_dir = f'zip_file/tempoary_log1208_before_rm'
os.makedirs(out_dir, exist_ok=True)
zip_source(f'{out_dir}//codes.zip')

for f_name in ['read_*']:
    os.system(f'cp {f_name} {out_dir}')

