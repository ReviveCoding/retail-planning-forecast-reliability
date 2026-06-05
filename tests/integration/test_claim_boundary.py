from src.utils import project_path


def test_claim_boundary_mentions_required_caveats_and_no_forbidden_claims():
    text = project_path('reports/claim_boundary.md').read_text(encoding='utf-8')
    assert 'synthetic/proxy planning assumptions' in text
    assert 'local/offline analytical framework' in text
    assert 'proprietary Amazon data' in text  # appears only as a negative/caveat statement
    assert 'not an Amazon internal system' in text
