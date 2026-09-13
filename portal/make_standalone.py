#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Builds ../docs/portal.html — the portal as one openable file.

The page is the served page with its back end moved inside: post() is
swapped for portal_local.js, which runs quantum.py and the server module
in Pyodide. Everything else — trees, annealing, the network views — is
already client-side.

    python3 make_standalone.py
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'docs', 'portal.html')


def read(name):
    with open(os.path.join(HERE, name), encoding='utf-8') as f:
        return f.read()


def main():
    page = read('quantum_portal.html')
    local_js = read('portal_local.js')

    post_re = re.compile(r'async function post\(url, body\) \{.*?\n\}\n',
                         re.S)
    if not post_re.search(page):
        raise SystemExit('post() not found')
    page = post_re.sub(lambda m: local_js, page, count=1)

    blocks = []
    for name in ('quantum.py', 'quantum_portal_server.py',
                 'portal_runtime.py'):
        ident = name.replace('.', '_')
        blocks.append('<script type="text/x-python" id="%s" data-file="%s">\n'
                      '%s</script>\n' % (ident, name, read(name)))
    page = page.replace('<script>\n', ''.join(blocks) + '<script>\n', 1)

    page = page.replace(
        '</h1>\n',
        '</h1>\n<div id="boot" class="hint" style="margin:-6px 0 10px">'
        'starting…</div>\n', 1)
    page = page.replace(
        '</script>\n</body>',
        "\n$('btnGo').disabled = true;\n"
        "pyReady = bootPython().catch(e => {\n"
        "  bootStatus(e.message, true);\n"
        "  throw e;\n"
        "});\n</script>\n</body>", 1)

    page = page.replace('<title>Quantum TNCO Portal</title>',
                        '<title>Quantum TNCO Portal (standalone)</title>', 1)

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(page)
    print('%s (%.0f KB)' % (os.path.abspath(OUT), os.path.getsize(OUT) / 1024))


if __name__ == '__main__':
    main()
