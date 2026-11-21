"""go_cli.proof_digests reads a mapped single-domain proof as the stage does.

A mapped route (aifs, icon-global, ecmwf-open-data, ...) records no top-level
``input_manifest_sha256``: its manifest is the evidence copy the forecast
runner re-hashes (gpuwm.stage_cli._source_manifest_digest).  proof_digests
required the top-level key and called such a proof "a multi-domain hierarchy
product".  Brought from jacobbieker/woof#72 (Apache-2.0).
"""
import hashlib
import json

import pytest

from gpuwm import go_cli, stage_cli


def _mapped_proof(tmp_path):
    root = tmp_path / "prep"
    (root / "source-evidence").mkdir(parents=True)
    manifest = root / "source-evidence" / "input-manifest.json"
    manifest.write_text('{"inputs": []}\n', encoding="utf-8")
    proof = root / "proof.json"
    proof.write_text(json.dumps({"prepared_cache": {"content_sha256": "c" * 64}}),
                     encoding="utf-8")
    return root, proof, manifest


def test_a_mapped_single_domain_proof_gives_the_stage_triple(tmp_path):
    root, proof, manifest = _mapped_proof(tmp_path)
    digests = go_cli.proof_digests(root)
    assert digests == {
        "proof": hashlib.sha256(proof.read_bytes()).hexdigest(),
        "source_manifest": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "prepared_content": "c" * 64}
    bundle = {"document": proof, "payload": json.loads(proof.read_text()), "layout": "single"}
    assert digests == stage_cli.single_domain_digests(bundle)


def test_a_proof_with_no_prepared_cache_is_still_refused(tmp_path):
    root = tmp_path / "prep"
    root.mkdir()
    (root / "proof.json").write_text("{}", encoding="utf-8")
    with pytest.raises(go_cli.GoRefusal, match="multi-domain hierarchy product"):
        go_cli.proof_digests(root)
