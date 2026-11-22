"""
scripts/train.py defaults = the frozen design (RESEARCH_PLAN.md P2.9, 19/11/2026), and the ablation switches.
Run from Code/:  python tests/test_train_cli.py      (pytest also works)
"""
import os
import sys
import tempfile

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CODE, "scripts"))
sys.path.insert(0, CODE)
import train  # noqa: E402


def test_defaults_are_the_frozen_design():
    a = train.parse_args(["--data", "x"])
    assert a.input_norm is True and a.text_proj is True and a.depth_dropout == 0.3
    assert a.depth_stats == "auto" and a.rel_gate is False


def test_ablation_switches():
    a = train.parse_args(["--data", "x", "--no-input-norm", "--no-text-proj", "--depth-dropout", "0",
                          "--depth-stats", "none"])
    assert a.input_norm is False and a.text_proj is False and a.depth_dropout == 0.0
    assert train.resolve_depth_stats(a.depth_stats, "x", "Spatial-VRD") is None
    a.depth_stats = None
    assert train.options_tag(a) == "base"


def test_auto_depth_stats_only_on_scene_graphs():
    with tempfile.TemporaryDirectory() as d:
        assert train.resolve_depth_stats("auto", d, "FB15k-237-IMG") is None   # knowledge graph: off
        try:
            train.resolve_depth_stats("auto", d, "Spatial-VRD")                # scene graph without the table
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("a scene graph without ent_depth_stats.pt must fail under auto")
        open(os.path.join(d, "ent_depth_stats.pt"), "wb").close()
        assert train.resolve_depth_stats("auto", d, "Spatial-VG") == "data"
        assert train.resolve_depth_stats("auto", d, "MKG-Y") is None


def test_tags():
    a = train.parse_args(["--data", "x"])
    a.depth_stats = "data"
    assert train.options_tag(a) == "inorm-ddrop0.3-tproj-dstat"
    a.depth_stats = None
    assert train.options_tag(a) == "inorm-ddrop0.3-tproj"


def test_budget_and_tune_names():
    assert train.EPOCH_BUDGET["MKG-Y"] == (1000, 50) and train.EPOCH_BUDGET["FB15k-237-IMG"] == (200, 10)
    a = train.parse_args(["--data", "x", "--tune", "--lr", "3e-4", "--dropout", "0.3"])
    assert a.epochs is None and a.valid_every is None                  # filled from the budget in main()
    assert train.config_tag(a) == "lr0.0003-do0.3"


def test_tune_summary_choice_and_convergence_rule():
    sys.path.insert(0, os.path.join(CODE, "scripts"))
    import tune_summary

    def run(v, lr, do, mrr, best, epochs=100):
        return {"variant": v, "config": {"lr": lr, "dropout": do}, "valid_mrr": mrr, "best_epoch": best,
                "epochs": epochs, "options_tag": "t", "file": f"{v}/{lr}-{do}"}
    runs = [run("full", 1e-4, 0.1, 0.20, 30), run("full", 3e-4, 0.1, 0.25, 95), run("full", 1e-3, 0.3, 0.25, 40),
            run("no_depth", 1e-4, 0.1, 0.30, 50)]
    c = tune_summary.choose(runs)
    assert c["full"]["config"] == {"lr": 3e-4, "dropout": 0.1}          # tie on MRR -> lower lr
    assert c["full"]["late_best_epoch"] and c["full"]["action"] == "rerun with --epochs 200"
    assert c["no_depth"]["action"] == "converged" and c["full"]["n_configs"] == 3



def test_edge_rule_points():
    sys.path.insert(0, os.path.join(CODE, "scripts"))
    from tune_summary import edge_points
    base = [(lr, do) for lr in (1e-4, 3e-4, 1e-3) for do in (0.1, 0.3)]
    pts = lambda tested, chosen: sorted((p["lr"], p["dropout"]) for p in edge_points(tested, chosen))  # noqa: E731
    assert pts(base, (3e-4, 0.3)) == [(3e-4, 0.5)]                       # inside on lr, high edge on dropout
    assert pts(base, (1e-3, 0.1)) == [(1e-3, 0.0), (3e-3, 0.1)]           # two edges -> one point per axis
    assert pts(base + [(1e-4, 0.0)], (1e-4, 0.0)) == [(3e-5, 0.0)]        # dropout 0 is a natural bound
    deep = base + [(3e-5, 0.1), (1e-5, 0.1)]
    assert pts(deep, (1e-5, 0.1)) == [(1e-5, 0.0)]                         # lr already 2 steps below the grid
    assert pts(base + [(3e-3, 0.1), (1e-3, 0.0)], (1e-3, 0.1)) == []       # chosen inside -> final



def test_convergence_rerun_names_and_once():
    a = train.parse_args(["--data", "x", "--tune", "--lr", "1e-4", "--dropout", "0.1", "--epochs", "400"])
    assert train.config_tag(a, 200) == "lr0.0001-do0.1-ep400"            # rerun never collides with the budget run
    a.epochs = 200
    assert train.config_tag(a, 200) == "lr0.0001-do0.1"
    sys.path.insert(0, os.path.join(CODE, "scripts"))
    import tune_summary

    def run(mrr, best, epochs, lr=1e-4):
        return {"variant": "full", "config": {"lr": lr, "dropout": 0.1}, "valid_mrr": mrr, "best_epoch": best,
                "epochs": epochs, "options_tag": "t", "file": f"{lr}-{epochs}"}
    c = tune_summary.choose([run(0.32, 190, 200), run(0.30, 50, 200, lr=3e-4)])["full"]
    assert c["rerun"] == {"lr": 1e-4, "dropout": 0.1, "epochs": 400} and not c["final"]
    c = tune_summary.choose([run(0.32, 190, 200), run(0.33, 390, 400), run(0.30, 50, 200, lr=3e-4)])["full"]
    assert c["epochs"] == 400 and c["rerun"] is None                      # doubled once: the rule is not repeated
    assert c["late_best_epoch"] and c["action"].startswith("late best epoch after the doubled budget")
    c = tune_summary.choose([run(0.32, 190, 200), run(0.32, 190, 400), run(0.30, 50, 200, lr=3e-4)])["full"]
    assert c["epochs"] == 200 and not c["late_best_epoch"] and c["rerun"] is None   # the doubled run settled it
    assert c["action"].startswith("converged (doubled budget: best epoch 190 / 400")


def test_resume_state_must_match_the_training_arguments():
    a = train.parse_args(["--data", "/a/Spatial-VRD", "--tune", "--lr", "1e-4", "--out", "/x"])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "resume_s.pt")
        assert train.load_resume(path, a) is None                        # nothing to resume
        train.save_resume(path, {"epoch": 3, "resume_key": train.resume_key(a)})
        assert not os.path.exists(path + ".tmp")
        moved = train.parse_args(["--data", "/b/Spatial-VRD", "--tune", "--lr", "1e-4", "--out", "/y",
                                  "--eval-batch-size", "64"])
        assert train.load_resume(path, moved)["epoch"] == 3               # other machine / folders: same run
        other = train.parse_args(["--data", "/a/Spatial-VRD", "--tune", "--lr", "3e-4", "--out", "/x"])
        try:
            train.load_resume(path, other)
        except SystemExit as e:
            assert "lr" in str(e)
        else:
            raise AssertionError("a resume state of another configuration must be refused")
        other = train.parse_args(["--data", "/a/MKG-Y", "--tune", "--lr", "1e-4", "--out", "/x"])
        try:
            train.load_resume(path, other)
        except SystemExit as e:
            assert "dataset" in str(e)
        else:
            raise AssertionError("a resume state of another dataset must be refused")
        a.no_resume = True
        assert train.load_resume(path, a) is None


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all train CLI tests passed")
