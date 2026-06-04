# Gemma4 Setup — pip_overrides Recipe

The `unified` pixi environment inside the container ships transformers 4.57.6, which predates Gemma4 support. This guide creates a local `pip_overrides/transformers/` directory containing transformers 5.5.0 with compatibility patches for the container's older dependencies.

The pipeline stage (`Gemma4DirectCaptionStage`, `JudgeGemma4Base`) injects this directory into `sys.path` at runtime so transformers 5.5.0 takes priority without touching the container image.

---

## Step 1 — Download transformers 5.5.0 (on any machine with pip)

```bash
pip download transformers==5.5.0 --no-deps -d /tmp/tw
unzip /tmp/tw/transformers-5.5.0-py3-none-any.whl "transformers/*" -d /tmp/tw/extracted/
```

## Step 2 — Copy to your workspace

```bash
# Replace <workspace> with your local workspace path
scp -r /tmp/tw/extracted/transformers/ \
    <user>@<cluster>:<workspace>/pip_overrides/
```

Or on the cluster directly if you have internet access:
```bash
pip download transformers==5.5.0 --no-deps -d /tmp/tw
unzip /tmp/tw/transformers-5.5.0-py3-none-any.whl "transformers/*" -d <workspace>/pip_overrides/
```

## Step 3 — Apply compatibility patches

The container ships `huggingface_hub==0.36.0` and `tokenizers==0.22.2`. Eight files need patching:

### Patch 1 — `dependency_versions_table.py`
Relax version constraints that block older huggingface_hub/tokenizers:
```python
# Change:
"huggingface-hub": ">=1.5.0,<2.0",
"tokenizers": ">=0.21,<0.22",
# To:
"huggingface-hub": ">=0.30.0",
"tokenizers": ">=0.21",
```

### Patch 2 — `utils/hub.py`
Add a local `is_offline_mode()` definition (5.5.0 imports it from huggingface_hub, but 0.36.0 doesn't have it):
```python
# Add near the top, after imports:
def is_offline_mode() -> bool:
    from huggingface_hub.constants import HF_HUB_OFFLINE
    return bool(HF_HUB_OFFLINE)
```

### Patch 3 — Remove `is_offline_mode` from huggingface_hub imports (10 files)
These files import `is_offline_mode` from `huggingface_hub`, which fails on 0.36.0. In each file, remove it from the `from huggingface_hub import ...` line and add `from transformers.utils.hub import is_offline_mode` after the `from __future__` block.

Files to patch:
- `modeling_utils.py`
- `tokenization_utils_base.py`
- `feature_extraction_utils.py`
- `modelcard.py`
- `tokenization_utils_tokenizers.py`
- `image_processing_base.py`
- `dynamic_module_utils.py`
- `video_processing_utils.py`
- `pipelines/__init__.py`
- `processing_utils.py`

### Patch 4 — `tokenization_mistral_common.py`
Wrap `ReasoningEffort` import in try/except (container's `mistral_common` is too old):
```python
try:
    from mistral_common.tokens.tokenizers.reasoning import ReasoningEffort
except ImportError:
    class ReasoningEffort:  # type: ignore
        none = "none"
        high = "high"
```

### Patch 5 — `modeling_gguf_pytorch_utils.py`
GGUF is not installed in the container. Wrap the `.integrations` import and use `.get()` for dict access:
```python
try:
    from . import integrations
except ImportError:
    integrations = None

# Replace any direct dict access like:
#   gguf_scalar_types[...]
# with:
#   gguf_scalar_types.get(...)
```

### Patch 6 — `models/auto/tokenization_auto.py`
Two changes needed for the lazy module loader:
```python
# Add at the very top of the file:
from __future__ import annotations

# Wrap hasattr calls that can trigger ModuleNotFoundError from lazy loading:
try:
    found = hasattr(main_module, class_name)
except Exception:
    found = False
```

### Patch 7 — `utils/auto_docstring.py`
The `Gemma4AudioModel` decorator triggers an import chain that fails when `tokenizers` is not on `sys.path`. Make the decorator silently pass on any error:
```python
def auto_docstring_decorator(obj):
    try:
        # ... original decorator logic ...
    except Exception:
        return obj
    return obj
```

### Patch 8 — Remove `pip_overrides/transformers/__init__.py`
Delete this file entirely. The 5.5.0 `__init__.py` tries to re-export symbols that depend on the full 5.5.0 environment and raises `ImportError` against the container's base. The container's own `transformers/__init__.py` (from the pixi env) is used instead — `PYTHONPATH` override makes the individual model modules available while keeping the init from the installed version.

---

## Step 4 — Verify

Inside the container:
```bash
PYTHONPATH=/config/pip_overrides python -c "
import sys
sys.path.insert(0, '/opt/cosmos-curate/.pixi/envs/unified/lib/python3.12/site-packages')
sys.path.insert(0, '/config/pip_overrides')
from transformers.models.gemma4 import Gemma4ForConditionalGeneration
print('Gemma4 import OK')
"
```

---

## Runtime injection order

In `stage_setup()`, `sys.path` must be injected in this exact order:

```python
import sys
# 1. unified env site-packages — provides the `tokenizers` C extension
sys.path.insert(0, "/opt/cosmos-curate/.pixi/envs/unified/lib/python3.12/site-packages")
# 2. pip_overrides — transformers 5.5.0 takes priority over unified env's 4.57.6
sys.path.insert(0, "/config/pip_overrides")
```

**Why order matters:** transformers' `__init__.py` caches `is_tokenizers_available()` the first time it is imported. If `pip_overrides` is inserted before the unified env, `tokenizers` (a compiled C extension) is not yet importable, `is_tokenizers_available()` returns `False`, and `GemmaTokenizerFast` becomes a `_DummyObject`. Inserting the unified env first ensures `tokenizers` is found before transformers initialises.
