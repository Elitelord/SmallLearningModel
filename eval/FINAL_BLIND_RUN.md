# Final blind holdout run

The final model and decode were locked before opening `blind_v4r5`:

- adapter: v4r8
- temperature: 0
- seed: 0
- prompt: bare `Explain: {concept}`
- holdout: all 24 prompts in frozen order

The three frontier models use the full grade-3 prompt. GPT-5.6 SOL and Gemini 3.1
Pro use temperature 0; Claude Opus 4.8 uses provider-default decoding because its API
rejects the deprecated temperature field.

## Frontier blind result

| Model | Readability | Clean 3/2 | Accuracy-v2 | Overall-v2 |
|---|---:|---:|---:|---:|
| Gemini 3.1 Pro | 16/24 | 22/24 | 24/24 | **16/24** |
| Claude Opus 4.8 | 11/24 | 21/24 | 24/24 | **11/24** |
| GPT-5.6 SOL | 10/24 | 24/24 | 24/24 | **10/24** |

Raw artifacts are `litmus/frontier_blind_v4r5_outputs.json` and
`litmus/frontier_blind_v4r5_accuracy_v2.json`.

## One-time v4r8 Colab cell

Stop the long-running Gradio cell before running this cell on the T4 runtime.

```python
from google.colab import drive
from huggingface_hub import snapshot_download
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

repo = Path('/content/SmallLearningModel_demo')
if not repo.exists():
    subprocess.run([
        'git', 'clone', 'https://github.com/Elitelord/SmallLearningModel.git', str(repo)
    ], check=True)
os.chdir(repo)

adapter_dir = Path('/content/v4r8_adapter')
snapshot_download(
    repo_id='SAgarwal34/qwen3-4b-grade3-science-v4r8',
    local_dir=adapter_dir,
)

output_path = Path('eval/v4r8_decode_blind_v4r5.json')
top_path = Path('eval/v4r8_decode_blind_v4r5.top.json')
if output_path.exists() or top_path.exists():
    raise FileExistsError('Blind output already exists; refusing to rerun it')

command = [
    sys.executable, '-u', '-m', 'eval.tuned_sweep',
    '--adapter', str(adapter_dir),
    '--eval-key', 'blind_v4r5', '--final-eval',
    '--temperatures', '0', '--seeds', '0',
    '--top-settings', '1', '--out', str(output_path),
]
subprocess.run(command, check=True)

result = json.loads(output_path.read_text(encoding='utf-8'))
print(
    'v4r8 blind readability:',
    sum(record['readability_pass'] for record in result['records']),
    '/', len(result['records']),
)

drive.mount('/content/drive')
backup = Path('/content/drive/MyDrive/SmallLearningModel/v4r8')
backup.mkdir(parents=True, exist_ok=True)
for path in (output_path, top_path):
    shutil.copy2(path, backup / path.name)
print(f'blind outputs backed up to {backup}')
```

Copy both generated JSON files into local `eval/`, then run accuracy-v2 exactly once.
