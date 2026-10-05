import os
import ast
import json
import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS = [
    "01_train_full.ipynb",
    "02_oracle_retrain.ipynb",
    "03_standard_mu.ipynb",
    "04a_cmf_static.ipynb",
    "04b_cmf_posthoc.ipynb",
    "04c_cmf_adaptive.ipynb",
]

@pytest.mark.parametrize("nb_path", NOTEBOOKS)
def test_notebook_json_validity_and_format(nb_path):
    """Test that the notebook is valid JSON and valid nbformat v4."""
    assert os.path.exists(nb_path), f"Notebook file {nb_path} does not exist"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    assert nb is not None
    assert "cells" in nb
    assert len(nb.cells) > 0


@pytest.mark.parametrize("nb_path", NOTEBOOKS)
def test_notebook_python_syntax(nb_path):
    """Test that all code cells in the notebook parse cleanly as valid Python syntax."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert len(code_cells) > 0, f"No code cells found in {nb_path}"
    
    full_code = []
    for idx, cell in enumerate(code_cells):
        src = cell.source
        # Ignore IPython magic commands (like %matplotlib, %time, !) when checking syntax
        clean_lines = []
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("%") or stripped.startswith("!"):
                clean_lines.append("# " + line)
            else:
                clean_lines.append(line)
        cell_code = "\n".join(clean_lines)
        try:
            ast.parse(cell_code)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {nb_path} cell {idx + 1}:\n{e}\n\nCode:\n{cell_code}")
        full_code.append(cell_code)
    
    # Also verify entire combined script parses cleanly
    try:
        ast.parse("\n\n".join(full_code))
    except SyntaxError as e:
        pytest.fail(f"Syntax error in combined code of {nb_path}:\n{e}")


def test_core_modules_importable():
    """Verify that experiment_utils, data_loader, and all unlearning methods import correctly."""
    import experiment_utils
    import data_loader
    import unlearning_methods
    from unlearning_methods import (
        unlearn_random_label,
        unlearn_neggrad_plus,
        unlearn_salun,
        generate_salun_mask,
        unlearn_scrub,
        DistillKL,
        unlearn_tarun_unsir,
    )
    assert hasattr(experiment_utils, "build_resnet18_classifier")
    assert hasattr(experiment_utils, "evaluate_three_tier_metrics")
    assert hasattr(data_loader, "get_unlearning_dataloaders")
