"""Materialize pinned reference modules without changing the repository checkout."""
import argparse,hashlib,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PIN='0bffd558fbe82525636eccf7bd30ec791267783e'
BLOBS={'data.py':'b840b277254292d41d5fa24ee938ec6c231e2742','features.py':'ba162005e8d6a1c8acc1a1a43254c71dc6b3a547','models.py':'f17c1e029f67340a8a3dd6998d3c114d7b2f5060'}

def blob_sha(contents):
    return hashlib.sha1(b'blob '+str(len(contents)).encode()+b'\0'+contents).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);args=ap.parse_args()
    repository=Path(args.repo).resolve();destination=ROOT/'reference';destination.mkdir(exist_ok=True)
    (destination/'__init__.py').write_text('')
    for name,expected in BLOBS.items():
        try:
            contents=subprocess.check_output(['git','-C',str(repository),'show',f'{PIN}:src/{name}'],stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError,FileNotFoundError):
            # A source ZIP or shallow checkout is supported only if the source
            # bytes exactly match the evaluated pin.
            contents=(repository/'src'/name).read_bytes()
        if blob_sha(contents)!=expected:
            raise RuntimeError(f'{name} differs from the evaluated source. Fetch commit {PIN} before reproducing this study.')
        target=destination/name
        if not target.exists() or target.read_bytes()!=contents:
            import os
            temporary=target.with_name(target.name+f'.{os.getpid()}.tmp')
            temporary.write_bytes(contents);temporary.replace(target)
    print('Loaded the verified reference source.')

if __name__=='__main__':main()
