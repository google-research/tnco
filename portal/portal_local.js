// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// The standalone page's back end, in the page itself.
//
// The served version POSTs to a Python process; here Pyodide runs the same
// quantum.py and the server module in the browser, and both contraction
// trees are built by the page's own annealer, as always.

const PYODIDE_VERSION = '0.28.3';

let pyodide = null, pyReady = null;

function bootStatus(msg, bad) {
  const el = document.getElementById('boot');
  if (!el) return;
  el.textContent = msg || '';
  el.className = bad ? 'err' : 'hint';
}

async function bootPython() {
  bootStatus('loading Python (Pyodide, ~10 MB on first open)…');
  const url = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
  await new Promise((ok, bad) => {
    const s = document.createElement('script');
    s.src = url + 'pyodide.js';
    s.onload = ok;
    s.onerror = () => bad(new Error('could not load Pyodide from the CDN — ' +
      'this page needs the network the first time it is opened'));
    document.head.appendChild(s);
  });
  pyodide = await loadPyodide({ indexURL: url });
  for (const src of document.querySelectorAll('script[type="text/x-python"]'))
    pyodide.FS.writeFile(src.dataset.file, src.textContent);
  pyodide.runPython('import sys; sys.path.insert(0, "")');
  pyodide.runPython('import portal_runtime as P');
  bootStatus('');
  document.getElementById('btnGo').disabled = false;
  return pyodide;
}

// Drop-in replacement for the server call: same routes, same JSON.
async function post(url, body) {
  await pyReady;
  const api = pyodide.globals.get('P').api;
  const j = JSON.parse(api(url, JSON.stringify(body || {})));
  if (j.error) {
    if (j.trace) console.error(j.trace);
    throw new Error(j.error);
  }
  return j;
}
