import csv

from cvrp.runlog import FIELDS, append_run, next_run_id


def _read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_next_run_id_starts_at_01_and_increments(tmp_path):
    p = tmp_path / "runs.csv"
    assert next_run_id(p) == "Run-01"

    append_run(p, [{"method": "NN"}, {"method": "D"}], "Run-01", {"eval_set": "x"})
    assert next_run_id(p) == "Run-02"

    append_run(p, [{"method": "NN"}], "Run-07", {"eval_set": "x"})
    assert next_run_id(p) == "Run-08"


def test_append_writes_header_once_and_keeps_korean_note(tmp_path):
    p = tmp_path / "sub" / "runs.csv"
    append_run(p, [{"method": "NN", "mean_km": 1.5}], "Run-01", {"note": "기준선"})
    append_run(p, [{"method": "D", "mean_km": 1.2}], "Run-02", {"note": "2-opt 추가"})

    text = p.read_text(encoding="utf-8-sig")
    assert text.count("run_id,") == 1

    rows = _read(p)
    assert list(rows[0].keys()) == FIELDS
    assert [r["run_id"] for r in rows] == ["Run-01", "Run-02"]
    assert rows[1]["note"] == "2-opt 추가"
    assert rows[0]["date"]  # 자동으로 채워짐


def test_unknown_fields_are_ignored(tmp_path):
    p = tmp_path / "runs.csv"
    append_run(p, [{"method": "NN", "not_a_field": 1}], "Run-01", {})
    assert "not_a_field" not in _read(p)[0]
