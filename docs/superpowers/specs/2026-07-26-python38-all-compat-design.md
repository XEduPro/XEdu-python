# Python 3.8 and full-install compatibility design

## Goal

Make the XEdu-python 2.1 source branch installable on Python 3.8 without
changing the lightweight default installation. The documented full install
must include the Gradio UI used by `XEdu.LLM.Client.run()`.

## Packaging

- Change the project Python floor from 3.9 to 3.8 and advertise Python 3.8.
- Preserve the current NumPy and Matplotlib ranges for Python 3.9 and newer.
- Select NumPy 1.24 and Matplotlib 3.7 on Python 3.8 because the current
  minimum versions require Python 3.9.
- Add an `llm` optional dependency containing `gradio>=4,<5`. Gradio 4 supports
  Python 3.8; Gradio 5 does not.
- Include the same Gradio range in the existing `all` extra. Do not add Gradio
  to core dependencies, so `pip install XEdu-python` remains lightweight.

## Notebook

Replace `importlib.resources.files`, which is unavailable on Python 3.8, with
resource discovery based on the installed `XEdu.examples` package path. Keep
the existing preference for a repository-local wheel and the fallback to
`XEdu-python[all]`.

## Documentation

Document the optional LLM/full install command and state that the 2.1 branch
supports Python 3.8+. Publishing a wheel or PyPI release is outside this pull
request; until that release exists, the branch remains a source distribution
target rather than a public package-index release.

## Verification

- Add packaging tests for the Python floor, environment-specific scientific
  dependencies, and Gradio's membership in `llm` and `all` only.
- Update the Notebook regression test to reject `importlib.resources.files`
  and require Python 3.8-compatible package-path discovery.
- Run the focused tests before and after implementation to demonstrate the
  red-green cycle.
- Resolve base and `all` dependencies for Python 3.8 and the current Python.
- Compile the package with Python 3.8, run the full non-slow test suite, build
  the wheel, and inspect its metadata and bundled Notebook.

## Non-goals

- Publishing to PyPI or attaching a wheel to a GitHub Release.
- Adding Gradio to the default dependency set.
- Changing inference behavior or model assets.
