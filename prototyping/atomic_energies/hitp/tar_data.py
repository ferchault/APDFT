import tarfile
import os.path
import sys

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

paths = []
with open(sys.argv[1], 'r') as f:
    for line in f:
        paths.append(line.strip('\n'))

for p in paths:
    print(f'Generating {p}.tar.gz', flush=True)
    make_tarfile(f'{p}.tar.gz', p)
