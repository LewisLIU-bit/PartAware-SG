"""Keep Florence dependencies isolated from RAM, GroundingDINO, and VLPart."""
import atexit
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys


class FlorenceWorker:
    def __init__(self, python, model_dir, device, max_tokens):
        self.device = device
        self.model = self
        self.process = subprocess.Popen(
            [str(python), '-u', str(Path(__file__).resolve()), '--worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8')
        atexit.register(self.close)
        self._request({'op': 'init', 'model_dir': str(model_dir),
                       'device': device, 'max_tokens': max_tokens})

    def _request(self, payload):
        if self.process.poll() is not None:
            raise RuntimeError(f'Florence worker exited: {self.process.returncode}')
        self.process.stdin.write(json.dumps(payload) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Florence worker closed its response stream; inspect stderr')
        reply = json.loads(line)
        if 'error' in reply:
            raise RuntimeError(f"Florence worker: {reply['error']}")
        return reply.get('result')

    def to(self, device):
        self._request({'op': 'device', 'device': str(device)})
        return self

    def score(self, image, descriptions):
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return self._request({'op': 'score', 'image': base64.b64encode(buffer.getvalue()).decode('ascii'),
                              'descriptions': descriptions})

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)


def make_florence_scorer(model_dir, device='cuda', max_tokens=96):
    python = os.environ.get('FLORENCE_PYTHON')
    if not python:
        from importlib.metadata import version, PackageNotFoundError
        try:
            compatible = int(version('timm').split('.')[0]) >= 1 and int(version('transformers').split('.')[1]) >= 49
            version('einops')
        except PackageNotFoundError:
            compatible = False
        if not compatible:
            candidate = Path.home() / 'miniconda3/envs/sg-florence/bin/python'
            if candidate.is_file():
                python = candidate
            else:
                raise RuntimeError('Florence dependencies are unavailable. Set FLORENCE_PYTHON to an isolated '
                                   'environment with transformers 4.49, timm 1.0 and einops 0.8.')
    if python and Path(python).resolve() != Path(sys.executable).resolve():
        return FlorenceWorker(python, model_dir, device, max_tokens)
    from .description_scoring import FlorenceDescriptionScorer
    return FlorenceDescriptionScorer(model_dir, device, max_tokens)


def run_worker():
    from contextlib import redirect_stdout
    from PIL import Image
    scorer = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                if request['op'] == 'init':
                    from description_scoring import FlorenceDescriptionScorer
                    scorer = FlorenceDescriptionScorer(request['model_dir'], request['device'], request['max_tokens'])
                    result = None
                elif request['op'] == 'device':
                    scorer.model.to(request['device'])
                    if request['device'] == 'cpu':
                        scorer.torch.cuda.empty_cache()
                    result = None
                elif request['op'] == 'score':
                    with Image.open(io.BytesIO(base64.b64decode(request['image']))) as image:
                        result = scorer.score(image.convert('RGB'), request['descriptions'])
                else:
                    raise ValueError('Unknown Florence worker operation')
            reply = {'result': result}
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            reply = {'error': f'{type(exc).__name__}: {exc}'}
        print(json.dumps(reply), flush=True)


if __name__ == '__main__':
    run_worker()
