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

for p in paths[:5]:
    print(f'{p[:-1]}.tar.gz')
    make_tarfile(f'{p[:-1]}.tar.gz', p[:-1])