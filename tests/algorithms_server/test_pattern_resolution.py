import pytest

from server_app import pattern_resolution


def test_extract_month_key_handles_plain_and_iso_dates():
    assert pattern_resolution.extract_month_key('2021-05-17') == '2021-05'
    assert pattern_resolution.extract_month_key('2021-05-17T12:30:00Z') == '2021-05'
    assert pattern_resolution.extract_month_key(None) is None
    assert pattern_resolution.extract_month_key('not-a-date') is None


def test_resolve_patterns_file_for_request_exact_or_raise_then_latest_default(tmp_path, monkeypatch):
    data_dir = tmp_path / 'data'
    md_dir = data_dir / 'patterns' / 'MD'
    md_dir.mkdir(parents=True)
    exact = md_dir / '2021-05-MD.parquet'
    exact.write_text('x', encoding='utf-8')
    earlier = md_dir / '2021-03-MD.parquet'
    earlier.write_text('x', encoding='utf-8')
    later = md_dir / '2021-08-MD.parquet'
    later.write_text('x', encoding='utf-8')

    monkeypatch.setattr(pattern_resolution, 'DATA_DIR', str(data_dir))

    exact_path, exact_source, exact_month = pattern_resolution.resolve_patterns_file_for_request(
        '240010001001',
        start_date_raw='2021-05-12',
        use_test_data=False,
    )
    assert exact_path == str(exact)
    assert exact_source == 'monthly'
    assert exact_month == '2021-05'

    # A requested month with no exact file must FAIL LOUDLY, not silently fall
    # back to the closest available month (2021-03 here) — that would corrupt
    # the simulation's mobility with no error. `earlier`/`later` exist on disk to
    # prove the resolver refuses to substitute them.
    with pytest.raises(ValueError):
        pattern_resolution.resolve_patterns_file_for_request(
            '240010001001',
            start_date_raw='2021-04-12',
            use_test_data=False,
        )

    latest_path, _, latest_month = pattern_resolution.resolve_patterns_file_for_request(
        '240010001001',
        start_date_raw=None,
        use_test_data=False,
    )
    assert latest_path == str(later)
    assert latest_month == '2021-08'
