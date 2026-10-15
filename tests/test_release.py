"""Verify the standalone installation wheel as well as archived content."""
import hashlib
import io
from contextlib import redirect_stdout
from pathlib import Path
import runpy
import tempfile
import unittest
import zipfile


class ReleaseTests(unittest.TestCase):
    def test_standalone_wheel_must_match_verified_archive(self):
        script = Path(__file__).resolve().parents[1]/'scripts/verify_releases.py'
        main = runpy.run_path(str(script))['main']
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); release = root/'releases'; release.mkdir()
            wheel = 'etissot-0.2.0-py3-none-any.whl'
            for name, prefix, members in [
                ('etissot-0.2.0.zip', 'etissot-0.2.0', {'dist/'+wheel: b'original wheel'}),
                ('etissot-manuscript-tex.zip', 'etissot-manuscript', {})]:
                path = release/name
                manifest = ''.join(hashlib.sha256(data).hexdigest()+'  '+member+'\n' for member, data in members.items())
                with zipfile.ZipFile(path, 'w') as archive:
                    archive.writestr(prefix+'/SHA256SUMS.txt', manifest)
                    for member, data in members.items():
                        archive.writestr(prefix+'/'+member, data)
                Path(str(path)+'.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
            main.__globals__['ROOT'] = root
            (release/wheel).write_bytes(b'changed wheel')
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(SystemExit, 'wheel differs'):
                main()
            (release/wheel).write_bytes(b'original wheel')
            with redirect_stdout(io.StringIO()):
                main()


if __name__ == '__main__':
    unittest.main()
