#!/usr/bin/env bash

 set -euo pipefail

 PATTERN="${1:-laser_cholera-*}"
 uvx --with twine twine check "dist/${PATTERN}"
 uvx --with twine twine upload --skip-existing "dist/${PATTERN}"
