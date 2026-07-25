"""Tests for the ON1 plugin: which file gets culled, and which metadata reaches disk."""

from pathlib import Path
import json

import pytest

from models import CullResult, ImageMetrics
from on1_plugin.resolve import resolve_paths
from sidecars import WriteOptions, write_on1_sidecar, write_xmp_sidecar


def touch(folder: Path, *names: str):
    for name in names:
        (folder / name).write_bytes(b"not really a photograph")


def photos(resolved):
    return [r.photo.name for r in resolved]


# ---------------------------------------------------------------- resolving


def test_a_rendered_copy_resolves_to_the_original(tmp_path):
    """ON1's Send To hands over a TIFF; the NEF beside it is what we are culling."""
    touch(tmp_path, "DSC_0001.NEF", "DSC_0001.tif")
    resolved = resolve_paths([str(tmp_path / "DSC_0001.tif")])
    assert photos(resolved) == ["DSC_0001.NEF"]
    assert resolved[0].was_redirected


def test_the_original_keeps_its_real_capitalisation(tmp_path):
    """The filesystem answers to DSC_0001.nef; only the true name matches elsewhere."""
    touch(tmp_path, "DSC_0001.NEF", "DSC_0001.tif")
    resolved = resolve_paths([str(tmp_path / "DSC_0001.tif")])
    assert resolved[0].photo.name == "DSC_0001.NEF"


def test_a_photo_and_its_render_collapse_into_one_entry(tmp_path):
    touch(tmp_path, "DSC_0001.NEF", "DSC_0001.tif")
    resolved = resolve_paths([str(tmp_path / "DSC_0001.NEF"), str(tmp_path / "DSC_0001.tif")])
    assert photos(resolved) == ["DSC_0001.NEF"]
    assert not resolved[0].was_redirected


@pytest.mark.parametrize("render", ["DSC_0001-2.tif", "DSC_0001 copy.psd", "DSC_0001-Edit.tif"])
def test_derived_names_trace_back(tmp_path, render):
    touch(tmp_path, "DSC_0001.NEF", render)
    assert photos(resolve_paths([str(tmp_path / render)])) == ["DSC_0001.NEF"]


def test_a_lone_jpeg_is_its_own_original(tmp_path):
    """Half the world shoots straight to JPEG; only a RAW sibling makes one a render."""
    touch(tmp_path, "holiday.jpg")
    resolved = resolve_paths([str(tmp_path / "holiday.jpg")])
    assert photos(resolved) == ["holiday.jpg"]
    assert not resolved[0].was_redirected


def test_a_selected_sidecar_follows_itself_home(tmp_path):
    touch(tmp_path, "DSC_0001.NEF", "DSC_0001.on1")
    assert photos(resolve_paths([str(tmp_path / "DSC_0001.on1")])) == ["DSC_0001.NEF"]


def test_folders_expand_to_originals_only(tmp_path):
    touch(tmp_path, "a.NEF", "a.tif", "a.on1", "b.jpg", "notes.txt")
    assert photos(resolve_paths([str(tmp_path)])) == ["a.NEF", "b.jpg"]


def test_unreadable_formats_are_dropped(tmp_path):
    touch(tmp_path, "clip.MOV", "notes.txt")
    assert resolve_paths([str(tmp_path)]) == []


# ---------------------------------------------------------------- writing


def make_result(photo: Path) -> CullResult:
    metrics = ImageMetrics(
        blur_score=0.9, exposure_score=0.9, composition_score=0.9, overall_quality=0.9,
        keywords=["seawall", "fog"], description="A bridge in fog",
        subject="bridge", subject_sharpness="sharp", exposure="good", framing="strong",
    )
    return CullResult(filepath=photo, decision="Keep", confidence=0.88, metrics=metrics)


def make_on1(photo: Path, metadata: dict):
    photo.with_suffix(".on1").write_text(
        json.dumps({"photos": {"id": {"name": photo.name, "metadata": metadata}}})
    )


def read_on1(photo: Path) -> dict:
    data = json.loads(photo.with_suffix(".on1").read_text())
    return next(iter(data["photos"].values()))["metadata"]


def test_unticked_fields_stay_out_of_the_on1_sidecar(tmp_path):
    photo = tmp_path / "a.NEF"
    make_on1(photo, {})
    write_on1_sidecar(
        make_result(photo),
        options=WriteOptions(keywords=False, description=False, analysis=False),
    )
    metadata = read_on1(photo)
    assert "seawall" not in metadata["Keywords"]
    assert "PhotoCuller:Keep" in metadata["Keywords"]
    assert "Description" not in metadata
    assert "PhotoCullerAnalysis" not in metadata


def test_with_no_keywords_ticked_the_field_is_left_alone(tmp_path):
    photo = tmp_path / "a.NEF"
    make_on1(photo, {"Keywords": ["Vancouver"]})
    write_on1_sidecar(
        make_result(photo), options=WriteOptions(keywords=False, culler_keywords=False)
    )
    assert read_on1(photo)["Keywords"] == ["Vancouver"]


def test_keywords_are_not_duplicated_across_runs(tmp_path):
    """Preserve mode keeps yours and appends ours, so a second run used to double up."""
    photo = tmp_path / "a.NEF"
    make_on1(photo, {})
    write_on1_sidecar(make_result(photo))
    write_on1_sidecar(make_result(photo))
    keywords = read_on1(photo)["Keywords"]
    assert keywords.count("seawall") == 1
    assert keywords.count("PhotoCuller:Keep") == 1


def test_case_only_duplicates_are_folded_together(tmp_path):
    photo = tmp_path / "a.NEF"
    make_on1(photo, {"Keywords": ["Seawall"]})
    write_on1_sidecar(make_result(photo))
    keywords = read_on1(photo)["Keywords"]
    assert keywords.count("Seawall") == 1
    assert "seawall" not in keywords


def test_a_ticked_rating_replaces_one_that_is_already_there(tmp_path):
    """The popup shows the photographer the rating before they tick it; theirs wins."""
    photo = tmp_path / "a.NEF"
    make_on1(photo, {"Rating": 2})
    write_on1_sidecar(make_result(photo), options=WriteOptions(rating=5, force_rating=True))
    assert read_on1(photo)["Rating"] == 5


def test_an_unticked_rating_is_never_touched(tmp_path):
    photo = tmp_path / "a.NEF"
    make_on1(photo, {"Rating": 2})
    write_on1_sidecar(make_result(photo))
    assert read_on1(photo)["Rating"] == 2


def test_a_ticked_description_replaces_the_existing_one(tmp_path):
    photo = tmp_path / "a.NEF"
    make_on1(photo, {"Description": "written last year"})
    write_on1_sidecar(make_result(photo), options=WriteOptions(force_description=True))
    assert read_on1(photo)["Description"] == "A bridge in fog"


def test_the_command_line_still_preserves_an_existing_description(tmp_path):
    photo = tmp_path / "a.NEF"
    make_on1(photo, {"Description": "written last year"})
    write_on1_sidecar(make_result(photo))
    assert read_on1(photo)["Description"] == "written last year"


def test_an_untouched_on1_sidecar_reports_failure(tmp_path):
    """No .on1 means ON1 has never seen the photo, and inventing one confuses it."""
    assert write_on1_sidecar(make_result(tmp_path / "a.NEF")) is False


def test_unticked_analysis_leaves_the_xmp_free_of_it(tmp_path):
    photo = tmp_path / "a.NEF"
    write_xmp_sidecar(make_result(photo), options=WriteOptions(analysis=False))
    content = photo.with_suffix(".xmp").read_text()
    assert "photoculler:decision" not in content
    assert "photoshop:Instructions" not in content
    assert "<dc:subject>" in content


def test_the_xmp_still_parses_with_everything_switched_off(tmp_path):
    import xml.etree.ElementTree as ET

    photo = tmp_path / "a.NEF"
    write_xmp_sidecar(
        make_result(photo),
        options=WriteOptions(
            keywords=False, culler_keywords=False, description=False, analysis=False
        ),
    )
    ET.parse(photo.with_suffix(".xmp"))
