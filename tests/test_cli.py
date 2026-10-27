from eegmodels.cli import main


def test_list_runs(capsys, tmp_path):
    assert main(["--cache", str(tmp_path), "list"]) == 0
    out = capsys.readouterr().out
    assert "NAME" in out and "steegformer-base" in out


def test_info_runs(capsys, tmp_path):
    assert main(["--cache", str(tmp_path), "info", "steegformer-base"]) == 0
    assert "braindecode/STEEGFormer-base" in capsys.readouterr().out


def test_unknown_model_is_error(capsys, tmp_path):
    assert main(["--cache", str(tmp_path), "info", "nope"]) == 1
    assert "Unknown model" in capsys.readouterr().err
