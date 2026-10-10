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

def test_unlearning_methods_smoke():
    """Verify that all unlearning methods run a fast forward/backward pass without errors."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from unlearning_methods import (
        unlearn_random_label,
        unlearn_neggrad_plus,
        unlearn_salun,
        unlearn_scrub,
        unlearn_tarun_unsir,
    )

    device = "cpu"
    # Create simple dummy model and datasets
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 16),
        nn.ReLU(),
        nn.Linear(16, 10),
    )
    
    # Create dummy retain and forget loaders
    x_retain = torch.randn(16, 3, 32, 32)
    y_retain = torch.randint(1, 10, (16,))
    retain_loader = DataLoader(TensorDataset(x_retain, y_retain), batch_size=8)

    x_forget = torch.randn(8, 3, 32, 32)
    y_forget = torch.zeros(8, dtype=torch.long)
    forget_loader = DataLoader(TensorDataset(x_forget, y_forget), batch_size=4)

    # 1. Random Label
    m1 = unlearn_random_label(
        model=nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 16), nn.ReLU(), nn.Linear(16, 10)),
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        num_classes=10,
        epochs=1,
        device=device,
    )
    assert m1 is not None

    # 2. SalUn
    m2 = unlearn_salun(
        model=nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 16), nn.ReLU(), nn.Linear(16, 10)),
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        num_classes=10,
        threshold=0.5,
        epochs=1,
        device=device,
    )
    assert m2 is not None

    # 3. SCRUB
    m3 = unlearn_scrub(
        model=nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 16), nn.ReLU(), nn.Linear(16, 10)),
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        epochs=1,
        msteps=1,
        device=device,
    )
    assert m3 is not None

    # 4. NegGrad+
    m4 = unlearn_neggrad_plus(
        model=nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 16), nn.ReLU(), nn.Linear(16, 10)),
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        epochs=1,
        device=device,
    )
    assert m4 is not None

    # 5. UNSIR / TarUn
    m5 = unlearn_tarun_unsir(
        model=nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 16), nn.ReLU(), nn.Linear(16, 10)),
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        forget_classes=[0],
        num_classes=10,
        img_shape=(3, 32, 32),
        noise_epochs=1,
        noise_steps=1,
        impair_epochs=1,
        impair_batches=1,
        repair_epochs=1,
        device=device,
    )
    assert m5 is not None

def test_visualization_utility_smoke():
    """Verify that visualize_feature_space_and_boundaries runs without errors."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from experiment_utils import FullClassifier, ClassifierHead, visualize_feature_space_and_boundaries

    class DummyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3 * 32 * 32, 16)
            self.feature_dim = 16
        def forward(self, x):
            return self.linear(x.view(x.size(0), -1))

    encoder = DummyEncoder()
    head = ClassifierHead(in_features=16, num_classes=10)
    model = FullClassifier(encoder, head)

    x_retain = torch.randn(20, 3, 32, 32)
    y_retain = torch.randint(0, 10, (20,))
    retain_loader = DataLoader(TensorDataset(x_retain, y_retain), batch_size=10)

    x_forget = torch.randn(10, 3, 32, 32)
    y_forget = torch.zeros(10, dtype=torch.long)
    forget_loader = DataLoader(TensorDataset(x_forget, y_forget), batch_size=5)

    visualize_feature_space_and_boundaries(
        model=model,
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        num_classes=10,
        forget_class=0,
        title="Test Visualization",
        device="cpu",
        save_path="./artifacts/plots/test_plot.png"
    )
    assert os.path.exists("./artifacts/plots/test_plot.png")

def test_summary_table_smoke():
    """Verify that print_metrics_summary_table formats and outputs correctly."""
    from experiment_utils import print_metrics_summary_table

    sample_results = [
        {
            "method": "random_label",
            "model_name": "resnet18",
            "seed": 0,
            "runtime_sec": 12.34,
            "test_metrics": {
                "output_retain": 0.92,
                "output_forget": 0.05,
                "lp_retain": 0.91,
                "lp_forget": 0.45,
                "illusion_gap_lp": 0.40,
                "ncc_retain": 0.89,
                "ncc_forget": 0.42,
                "illusion_gap_ncc": 0.37,
            }
        }
    ]
    df = print_metrics_summary_table(
        results=sample_results,
        title="Test Summary",
        save_csv_path="./artifacts/metrics/test_summary.csv"
    )
    assert df is not None
    assert len(df) == 1
    assert os.path.exists("./artifacts/metrics/test_summary.csv")
def test_evaluate_three_tier_metrics_dataset_separation():
    """Verify that evaluate_three_tier_metrics handles probe training, eval, and test correctly."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from experiment_utils import FullClassifier, ClassifierHead, evaluate_three_tier_metrics

    class DummyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3 * 32 * 32, 16)
            self.feature_dim = 16
        def forward(self, x):
            return self.linear(x.view(x.size(0), -1))

    encoder = DummyEncoder()
    head = ClassifierHead(in_features=16, num_classes=10)
    model = FullClassifier(encoder, head)

    x_r = torch.randn(20, 3, 32, 32)
    y_r = torch.randint(0, 10, (20,))
    r_loader = DataLoader(TensorDataset(x_r, y_r), batch_size=10)

    x_f = torch.randn(10, 3, 32, 32)
    y_f = torch.randint(0, 10, (10,))
    f_loader = DataLoader(TensorDataset(x_f, y_f), batch_size=5)

    x_t = torch.randn(10, 3, 32, 32)
    y_t = torch.randint(0, 10, (10,))
    t_loader = DataLoader(TensorDataset(x_t, y_t), batch_size=5)

    res = evaluate_three_tier_metrics(
        model=model,
        probe_train_loader=r_loader,
        retain_eval_loader=r_loader,
        forget_eval_loader=f_loader,
        num_classes=10,
        seed=123,
        device="cpu",
        test_eval_loader=t_loader,
        nonmember_loader=f_loader,  # use forget loader as stand-in non-member set for the smoke test
    )
    assert "output_retain" in res
    assert "output_forget" in res
    assert "output_test" in res
    assert "lp_retain" in res
    assert "lp_forget" in res
    assert "lp_test" in res
    assert "ncc_retain" in res
    assert "ncc_forget" in res
    assert "ncc_test" in res
    assert "illusion_gap_lp" in res
    assert "illusion_gap_ncc" in res
    assert "mia_forget" in res
    assert "disc_forget" in res
    assert "indisc_forget" in res


def test_mia_computation_smoke():
    """Verify compute_mia_metrics calculates AUC, Discernibility, and Indiscernibility."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from experiment_utils import compute_mia_metrics

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 10)
    )

    x_f = torch.randn(20, 3, 32, 32)
    y_f = torch.randint(0, 10, (20,))
    f_loader = DataLoader(TensorDataset(x_f, y_f), batch_size=10)

    x_nm = torch.randn(20, 3, 32, 32)
    y_nm = torch.randint(0, 10, (20,))
    nm_loader = DataLoader(TensorDataset(x_nm, y_nm), batch_size=10)

    mia_res = compute_mia_metrics(model, f_loader, nm_loader, device="cpu")
    assert "mia_forget" in mia_res
    assert "disc_forget" in mia_res
    assert "indisc_forget" in mia_res
    assert 0.0 <= mia_res["mia_forget"] <= 1.0
    assert 0.0 <= mia_res["disc_forget"] <= 1.0
    assert 0.0 <= mia_res["indisc_forget"] <= 1.0


def test_class_mean_geometry_vs_oracle():
    """Geometry comparison is rotation-invariant: a rotated copy of the reference matches it exactly."""
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from experiment_utils import (
        FullClassifier, ClassifierHead,
        compute_class_mean_geometry, compare_class_mean_geometry,
        save_class_mean_geometry, load_oracle_geometry, print_geometry_summary_table,
    )

    torch.manual_seed(0)
    D, C = 16, 5

    class LinEncoder(nn.Module):
        def __init__(self, W):
            super().__init__()
            self.W = W
            self.feature_dim = D
        def forward(self, x):
            return x.view(x.size(0), -1) @ self.W

    W_ref = torch.randn(3 * 32 * 32, D)
    Q, _ = torch.linalg.qr(torch.randn(D, D))  # random orthogonal rotation of the feature basis
    ref_model = FullClassifier(LinEncoder(W_ref), ClassifierHead(D, C))
    rot_model = FullClassifier(LinEncoder(W_ref @ Q), ClassifierHead(D, C))

    x_r, y_r = torch.randn(50, 3, 32, 32), torch.arange(50) % C
    x_f, y_f = torch.randn(20, 3, 32, 32), torch.arange(20) % C
    r_loader = DataLoader(TensorDataset(x_r, y_r), batch_size=10)
    f_loader = DataLoader(TensorDataset(x_f, y_f), batch_size=10)

    ref_geom = compute_class_mean_geometry(ref_model, r_loader, f_loader, num_classes=C, device="cpu")
    assert ref_geom["retain_directions"].shape == (C, D)
    assert np.allclose(ref_geom["retain_directions"].mean(axis=0), 0.0, atol=1e-6)

    save_class_mean_geometry(ref_geom, "./artifacts/metrics/oracle_geometry_testnet_seed_0.npz")
    loaded = load_oracle_geometry("testnet", 0, search_dirs=("./artifacts/metrics",))

    rot_geom = compute_class_mean_geometry(rot_model, r_loader, f_loader, num_classes=C, device="cpu")
    cmp = compare_class_mean_geometry(rot_geom, loaded)
    assert abs(cmp["retain"]["cos_mean"] - 1.0) < 1e-4
    assert abs(cmp["retain"]["norm_ratio_mean"] - 1.0) < 1e-4
    assert cmp["retain"]["procrustes_disparity"] < 1e-6
    assert cmp["retain"]["gram_cos_mae"] < 1e-5
    assert "forget" in cmp and len(cmp["forget"]["per_class_cos"]) == C

    # A different encoder must deviate from the reference.
    other = FullClassifier(LinEncoder(torch.randn(3 * 32 * 32, D)), ClassifierHead(D, C))
    cmp_other = compare_class_mean_geometry(
        compute_class_mean_geometry(other, r_loader, f_loader, num_classes=C, device="cpu"), loaded
    )
    assert cmp_other["retain"]["procrustes_disparity"] > cmp["retain"]["procrustes_disparity"]

    df = print_geometry_summary_table(
        [
            {"method": "rotated", "model_name": "testnet", "seed": 0, "geometry_vs_oracle": cmp},
            {"method": "other", "model_name": "testnet", "seed": 0, "geometry_vs_oracle": cmp_other},
        ],
        save_csv_path="./artifacts/metrics/test_geometry_summary.csv",
    )
    assert len(df) == 2
    assert os.path.exists("./artifacts/metrics/test_geometry_summary_by_method.csv")


def test_cmf_head_matches_paper_algorithms():
    """CMF head = Algorithm 1 (zero bias, frozen) and is rebuilt after every epoch (Algorithm 2) for all methods."""
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from experiment_utils import FullClassifier, ClassifierHead, extract_features, reconstruct_cmf_head
    from unlearning_methods import (
        unlearn_random_label, unlearn_neggrad_plus, unlearn_salun, unlearn_scrub, unlearn_tarun_unsir,
    )

    C, D = 4, 8

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(3 * 32 * 32, D)
            self.feature_dim = D
        def forward(self, x):
            return torch.relu(self.l(x.view(x.size(0), -1)))

    x, y = torch.randn(24, 3, 32, 32), torch.arange(24) % C
    data_loader = DataLoader(TensorDataset(x, y), batch_size=8)
    r_loader = DataLoader(TensorDataset(x[:16], y[:16]), batch_size=8)
    f_loader = DataLoader(TensorDataset(x[16:], y[16:]), batch_size=8)

    def expected_head(model):
        f, t = extract_features(model.encoder, data_loader, device="cpu")
        mu = np.stack([f[t == c].mean(0) for c in range(C)])          # Eq. 3: class means
        mu_bar = mu.mean(0)                                            # Eq. 3: mean of class means
        return (mu - mu_bar) / np.linalg.norm(mu - mu_bar, axis=1, keepdims=True)  # Alg. 1 line 6

    torch.manual_seed(0)
    model = FullClassifier(Enc(), ClassifierHead(D, C))
    reconstruct_cmf_head(model, data_loader, num_classes=C, device="cpu")
    assert np.allclose(model.head.linear.weight.detach().numpy(), expected_head(model), atol=1e-5)
    assert torch.all(model.head.linear.bias == 0)
    assert not model.head.linear.weight.requires_grad and not model.head.linear.bias.requires_grad

    runs = {
        "random_label": lambda m, cb: unlearn_random_label(m, r_loader, f_loader, num_classes=C, lr=1e-2, epochs=2, device="cpu", epoch_end_callback=cb),
        "neggrad_plus": lambda m, cb: unlearn_neggrad_plus(m, r_loader, f_loader, lr=1e-2, epochs=2, device="cpu", epoch_end_callback=cb),
        "salun": lambda m, cb: unlearn_salun(m, r_loader, f_loader, num_classes=C, lr=1e-2, epochs=2, device="cpu", epoch_end_callback=cb),
        "scrub": lambda m, cb: unlearn_scrub(m, r_loader, f_loader, epochs=2, msteps=1, lr=1e-2, device="cpu", epoch_end_callback=cb),
        "tarun_unsir": lambda m, cb: unlearn_tarun_unsir(m, r_loader, f_loader, num_classes=C, noise_epochs=1, noise_steps=1, impair_batches=1, impair_lr=1e-2, repair_lr=1e-2, device="cpu", epoch_end_callback=cb),
    }
    for name, run in runs.items():
        torch.manual_seed(0)
        m = FullClassifier(Enc(), ClassifierHead(D, C))
        reconstruct_cmf_head(m, data_loader, num_classes=C, device="cpu")
        calls = []
        def cb(model_):
            reconstruct_cmf_head(model_, data_loader, num_classes=C, device="cpu")
            calls.append(1)
        out = run(m, cb)
        assert len(calls) == 2, f"{name}: expected 2 per-epoch reconstructions, got {len(calls)}"
        # Final head must equal Algorithm 1 applied to the final encoder.
        assert np.allclose(out.head.linear.weight.detach().numpy(), expected_head(out), atol=1e-5), name
        assert torch.all(out.head.linear.bias == 0), name
