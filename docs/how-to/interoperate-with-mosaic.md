# Interoperate with MOSAIC

`laser.cholera` was originally a Python port of [MOSAIC](https://github.com/InstituteforDiseaseModeling/MOSAIC), and MOSAIC remains the canonical R-side scenario authoring tool for the metapopulation cholera workflow. MOSAIC writes parameter sets as JSON files that `laser.cholera` consumes directly via `get_parameters(<path>)` — no glue code, no schema translation, no intermediate format.

## Use a MOSAIC-produced JSON

```python
from laser.cholera.metapop.params import get_parameters
from laser.cholera.metapop.model import run_model

params = get_parameters("/path/to/mosaic-output.json")  # or .json.gz
model = run_model(params)
```

`get_parameters` accepts a `str`, a `pathlib.Path`, `None` (loads the bundled defaults), or a `dict` (already-parsed parameters). MOSAIC drops a JSON file on disk; pass its path and you're done. The bundled defaults in `src/laser/cholera/metapop/data/default_parameters.json` were themselves produced from MOSAIC, so a MOSAIC export is the reference shape `laser.cholera` expects.

## MOSAIC documentation

The MOSAIC project documents its own JSON schema, scenario builder, and calibration workflow at <https://github.com/InstituteforDiseaseModeling/MOSAIC>. This page intentionally does not duplicate that content — go to the MOSAIC repository for the authoritative description of how scenarios are authored on the R side.

## See also

- [How-to: override parameters](override-parameters.md) — if you need to tweak individual fields after loading a MOSAIC JSON.
- [Parameter reference](../reference/parameters/index.md) — the authoritative source for what each field means in `laser.cholera`.
- [Usage](../usage.md) — the bundled default JSON is a MOSAIC output.
