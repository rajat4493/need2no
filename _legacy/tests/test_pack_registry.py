from n2n.packs import list_packs


def test_pack_registry_contains_expected_ids():
    packs = list_packs()
    assert "global.pci_lite.v1" in packs
    assert "uk.bank_statement.v1" in packs
