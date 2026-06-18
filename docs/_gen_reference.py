"""Generate one Markdown stub per Python module for mkdocstrings.

Replaces ``sphinx-apidoc``. Walks ``src/laser/cholera`` and writes one
``docs/reference/<module-path>.md`` file per module.

Each generated file has the form::

    ::: laser.cholera.metapop.params

The mkdocstrings handler renders the module heading itself via
``show_root_heading: true`` in ``mkdocs.yml``; writing a Markdown H1
here as well produced a duplicated heading on every reference page.

Files are emitted into the MkDocs ``docs_dir`` at build time via
``mkdocs_gen_files.open(...)``. They are derived from ``src/`` and should not
be committed (see the repo's ``.gitignore`` entries for ``docs/reference/``).

Nav for the generated subtree is hand-listed in ``mkdocs.yml`` (the
``literate-nav`` plugin was retired in favour of an explicit Diátaxis-shaped
nav block); this script no longer emits a ``SUMMARY.md``.

Run automatically by the ``gen-files`` plugin (see ``mkdocs.yml``); do
not invoke directly.
"""

from pathlib import Path

import mkdocs_gen_files

SRC_ROOT = Path("src/laser/cholera")
REFERENCE_ROOT = Path("reference")

# Walk every .py file under the package, skipping anything obviously
# generated or internal-to-the-build (cache files, etc.).
for path in sorted(SRC_ROOT.rglob("*.py")):
    module_path = path.relative_to("src").with_suffix("")
    doc_path = REFERENCE_ROOT / path.relative_to(SRC_ROOT).with_suffix(".md")
    parts = tuple(module_path.parts)

    # Treat __init__.py as the package's index page.
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        # Skip module-as-script entry points; mkdocstrings can't render them
        # in a useful way.
        continue

    identifier = ".".join(parts)

    with mkdocs_gen_files.open(doc_path, "w") as fd:
        fd.write(f"::: {identifier}\n")
