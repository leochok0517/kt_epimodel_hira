"""기본 설치 검증."""


def test_kt_data_importable() -> None:
    import kt_data
    assert kt_data.__version__ == "0.1.0"


def test_kt_epimodel_hira_importable() -> None:
    import kt_epimodel_hira
    assert kt_epimodel_hira.__version__ == "0.1.0"


def test_kt_data_loader_works() -> None:
    """kt_data 로더가 kt_epimodel_hira 환경에서 호출 가능한지."""
    from kt_data.data.load_population import load_population_15groups
    df = load_population_15groups()
    assert df.shape[0] > 0


def test_kt_data_hira_loader_constants() -> None:
    """kt_data 의 HIRA 상수가 노출되는지 (file I/O 없이)."""
    from kt_data import HIRA_AGE_GROUPS, SUDOGWON_SIDO_CODES
    assert HIRA_AGE_GROUPS == ["0-5", "6-11", "12-17", "18-44", "45-64", "65+"]
    assert SUDOGWON_SIDO_CODES == [11, 28, 41]
