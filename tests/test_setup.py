"""Run real batch control flow with only network/install commands stubbed."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


STUB = r"""
import json, os, sys
from pathlib import Path
kind, *args = sys.argv[1:]
root = Path(os.environ['CREEPTY_COMFY_DIR'])
if kind == 'pip':
    stage = 'pip_upgrade' if '--upgrade' in args else ('pip_requirements' if '-r' in args else 'pip_torch')
elif kind == 'curl':
    output = Path(args[args.index('--output') + 1])
    stage = 'curl_' + ('image' if 'flux-2-klein' in output.name else ('encoder' if 'qwen' in output.name else 'vae'))
else:
    stage = kind
with open(os.environ['CREEPTY_TEST_LOG'], 'a', encoding='utf-8') as log:
    log.write(json.dumps(stage) + '\n')
failed = stage == os.environ.get('CREEPTY_TEST_FAIL')
if kind == 'git' and not failed:
    root.mkdir(parents=True, exist_ok=True)
if kind == 'py' and not failed:
    python = root / 'venv/Scripts/python.exe'
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_bytes(b'stub')
if kind == 'curl':
    output.write_bytes(b'' if os.environ.get('CREEPTY_TEST_EMPTY') else b'model weights')
raise SystemExit(1 if failed else 0)
"""


@unittest.skipUnless(os.name == 'nt', 'Windows batch installer')
class SetupTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(prefix='creepty setup ')))
        self.comfy = self.root / 'Comfy UI'
        self.log = self.root / 'calls.jsonl'
        source = (Path(__file__).resolve().parents[1] / 'scripts/setup_comfyui.bat').read_text()
        source = source.replace('    git ', '    call "%CREEPTY_TEST_STUB%" git ')
        source = source.replace('    py ', '    call "%CREEPTY_TEST_STUB%" py ')
        source = source.replace('"%COMFY_PYTHON%" -m pip ', 'call "%CREEPTY_TEST_STUB%" pip ')
        source = source.replace('curl --', 'call "%CREEPTY_TEST_STUB%" curl --')
        self.script = self.root / 'setup test.bat'
        self.script.write_text(source, encoding='utf-8')
        (self.root / 'stub.py').write_text(STUB, encoding='utf-8')
        stub = self.root / 'stub.cmd'
        stub.write_text('@echo off\n"%CREEPTY_TEST_PYTHON%" -B "%~dp0stub.py" %*\nexit /b %errorlevel%\n')
        self.env = dict(os.environ, CREEPTY_COMFY_DIR=str(self.comfy), CREEPTY_TEST_STUB=str(stub),
                        CREEPTY_TEST_PYTHON=sys.executable, CREEPTY_TEST_LOG=str(self.log),
                        CREEPTY_TEST_FAIL='', CREEPTY_TEST_EMPTY='')

    def run_setup(self, fail='', empty=False):
        self.log.unlink(missing_ok=True)
        env = dict(self.env, CREEPTY_TEST_FAIL=fail, CREEPTY_TEST_EMPTY='1' if empty else '')
        result = subprocess.run(['cmd.exe', '/d', '/c', str(self.script)], cwd=self.root,
                                env=env, capture_output=True, text=True, timeout=20)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return result, calls

    def test_stops_at_each_failed_stage_and_removes_partial_download(self):
        stages = ['git', 'py', 'pip_upgrade', 'pip_torch', 'pip_requirements', 'curl_image', 'curl_encoder', 'curl_vae']
        for index, stage in enumerate(stages):
            with self.subTest(stage=stage):
                # Each iteration has an isolated fresh destination.
                self.env['CREEPTY_COMFY_DIR'] = str(self.root / stage / 'Comfy UI')
                result, calls = self.run_setup(fail=stage)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(calls, stages[:index + 1])
                self.assertNotIn('ComfyUI ready.', result.stdout)
                self.assertFalse(list((self.root / stage).rglob('*.part')))

    def test_success_publishes_models_and_second_run_skips_downloads(self):
        result, calls = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('ComfyUI ready.', result.stdout)
        models = list(self.comfy.rglob('*.safetensors'))
        self.assertEqual(len(models), 3)
        self.assertTrue(all(p.stat().st_size for p in models))
        self.assertFalse(list(self.comfy.rglob('*.part')))
        result, calls = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, ['git', 'pip_upgrade', 'pip_torch', 'pip_requirements'])

    def test_empty_existing_model_is_replaced(self):
        self.assertEqual(self.run_setup()[0].returncode, 0)
        target = next(self.comfy.rglob('flux-2-klein*.safetensors'))
        target.write_bytes(b'')
        result, calls = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('curl_image', calls)
        self.assertGreater(target.stat().st_size, 0)

    def test_empty_download_fails_without_publishing_model(self):
        result, calls = self.run_setup(empty=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls[-1], 'curl_image')
        self.assertNotIn('ComfyUI ready.', result.stdout)
        self.assertFalse(list(self.comfy.rglob('*.safetensors')))
        self.assertFalse(list(self.comfy.rglob('*.part')))
