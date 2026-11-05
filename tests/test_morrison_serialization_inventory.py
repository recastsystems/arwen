"""One shared restart inventory owns every optional aerosol carrier."""
from types import SimpleNamespace
from gpuwm import state_serialization_contract as contract
from gpuwm.io import restart
from gpuwm.core.morrison_aero import AERO_DIAGNOSTIC_FIELDS

def test_all_inventory_consumers_share_the_same_optional_fields():
    assert restart.STATE_SERIALIZED_ATTRS is contract.STATE_SERIALIZED_ATTRS
    assert contract.serialized_state_attrs(SimpleNamespace()) == restart.STATE_SERIALIZED_ATTRS
    for name in AERO_DIAGNOSTIC_FIELDS:
        assert contract.STATE_SERIALIZED_ATTRS.count(name) == 1
        assert restart.classify_state_attr(name) == "serialize"
