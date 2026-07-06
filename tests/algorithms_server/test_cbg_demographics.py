import csv
import sys
from pathlib import Path

_SERVER = Path(__file__).resolve().parents[2] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))
for _m in ("cbg_demographics", "patterns", "patterns_loader", "popgen"):
    sys.modules.pop(_m, None)

from cbg_demographics import B01001_SEX_AGE_BUCKETS, CbgSexAgeSampler  # noqa: E402
from popgen import SyntheticPopulationGenerator  # noqa: E402


CBG = "400010001001"


def _write_b01001(path: Path, counts: dict[str, int]) -> None:
    fields = ["census_block_group"] + [bucket.field for bucket in B01001_SEX_AGE_BUCKETS]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        row = {"census_block_group": CBG}
        for field in fields[1:]:
            row[field] = counts.get(field, 0)
        writer.writerow(row)


def test_cbg_sex_age_sampler_filters_by_age_range_and_sex(tmp_path):
    csv_path = tmp_path / "cbg_b01.csv"
    _write_b01001(csv_path, {
        "B01001e15": 12,  # male 45-49
        "B01001e29": 20,  # female 10-14
    })

    sampler = CbgSexAgeSampler.load(csv_path, cbgs=[CBG])

    assert sampler is not None
    sex, age = sampler.sample(CBG, 18, 95, sex="M")
    assert sex == "M"
    assert 45 <= age <= 49

    sex, age = sampler.sample(CBG, 0, 17)
    assert sex == "F"
    assert 10 <= age <= 14


def test_default_loader_uses_algorithms_data_cwd(tmp_path, monkeypatch):
    data_cwd = tmp_path / "Algorithms" / "server"
    csv_path = data_cwd / "data" / "cbg_b01.csv"
    csv_path.parent.mkdir(parents=True)
    _write_b01001(csv_path, {"B01001e15": 12})
    monkeypatch.delenv("DELINEO_CBG_DEMOGRAPHICS_PATH", raising=False)
    monkeypatch.setenv("DELINEO_ALGORITHMS_DATA_CWD", str(data_cwd))

    sampler = CbgSexAgeSampler.load_default([CBG])

    assert sampler is not None
    assert sampler.source_path == csv_path


def test_population_generator_uses_cbg_sex_age_when_available(tmp_path, monkeypatch):
    csv_path = tmp_path / "cbg_b01.csv"
    _write_b01001(csv_path, {
        "B01001e15": 12,  # male 45-49
        "B01001e29": 20,  # female 10-14
    })
    monkeypatch.setenv("DELINEO_CBG_DEMOGRAPHICS_PATH", str(csv_path))

    generator = SyntheticPopulationGenerator(
        census_data={"001": {}},
        cz_data={CBG: 100},
    )

    sex, age = generator.generate_demographics(CBG, "householder", gender="M")
    assert sex == "M"
    assert 45 <= age <= 49

    sex, age = generator.generate_demographics(CBG, "child")
    assert sex == "F"
    assert 10 <= age <= 14
