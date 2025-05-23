pip install --upgrade pip > /tmp/pip.stdout.log 2> /tmp/pip.stderr.log
pip install uv > /tmp/uv.stdout.log 2> /tmp/uv.stderr.log
uv tool install tox --with tox-uv > /tmp/tox.stdout.log 2> /tmp/tox.stderr.log
uv venv --python 3.10 > /tmp/venv.stdout.log 2> /tmp/venv.stderr.log
source .venv/bin/activate
uv pip install -e .[nb] > /tmp/install.stdout.log 2> /tmp/install.stderr.log
