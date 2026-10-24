import ast
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import MagicMock

import numpy as np

from study import export_results


class NotebookTests(unittest.TestCase):
    def test_study_directory_contains_only_selected_workflow(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual({path.name for path in root.glob("*.ipynb")}, {"selected_study.ipynb"})
        self.assertEqual({path.name for path in root.glob("*.py")},
                         {"config.py", "datasets.py", "models.py", "presentation.py",
                          "redundancy.py", "study.py", "training_process.py"})

    def test_cells_execute_with_smoke_fits_and_mocked_plotting(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "selected_study.ipynb").read_text())
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(len({cell["id"] for cell in notebook["cells"]}), len(notebook["cells"]))
        pyplot = MagicMock(name="pyplot")

        def subplots(rows, columns, **kwargs):
            axes = np.empty((rows, columns), dtype=object)
            for index in range(rows * columns):
                axes.flat[index] = MagicMock(name=f"axis_{index}")
            if kwargs.get("squeeze", True):
                axes = axes.squeeze()
            return MagicMock(name="figure"), axes

        pyplot.subplots.side_effect = subplots
        matplotlib = ModuleType("matplotlib")
        matplotlib.pyplot = pyplot
        ipython = ModuleType("IPython")
        display = ModuleType("IPython.display")
        display.Markdown = lambda text: text
        display.display = MagicMock(name="display")
        modules = {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot,
                   "IPython": ipython, "IPython.display": display}
        namespace = {"__name__": "__notebook__"}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack, redirect_stdout(io.StringIO()):
            for name, module in modules.items():
                if name in sys.modules:
                    stack.callback(sys.modules.__setitem__, name, sys.modules[name])
                else:
                    stack.callback(sys.modules.pop, name, None)
                sys.modules[name] = module
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                self.assertEqual(cell.get("outputs", []), [])
                self.assertIsNone(cell.get("execution_count"))
                ast.parse(source)
                exec(compile(source, "selected_study.ipynb", "exec"), namespace)
                if "RERUN = True" in source:
                    namespace["SMOKE"] = True
                    namespace["export_results"] = lambda bundle, output_root: export_results(bundle, directory)
            if "bundle" in namespace:
                self.assertEqual(namespace["bundle"]["protocol"], "smoke")
                self.assertEqual(len(namespace["bundle"]["cases"]), 7)
            if "output" in namespace:
                self.assertTrue((namespace["output"] / "results.json").is_file())
        self.assertEqual(pyplot.subplots.call_count >= 2, True)


if __name__ == "__main__":
    unittest.main()