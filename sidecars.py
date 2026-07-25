"""Writing culling results into photo-app metadata.

Two formats, one set of rules:

  ON1  - updates the .on1 JSON sidecar in place. ON1 has to have created the file first;
         we never invent one, because a sidecar ON1 did not write can confuse its
         catalogue more than missing metadata does.
  XMP  - writes a standard .xmp sidecar that Lightroom, Bridge, Capture One and ON1 all
         read.

Both preserve what the photographer put there. Ratings, user keywords and existing
descriptions survive unless --override is passed, and even then ratings are kept.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from xml.sax.saxutils import escape
import json
import logging
import xml.etree.ElementTree as ET

try:
    from .decision import suggested_rating
    from .models import CullResult
except ImportError:
    from decision import suggested_rating
    from models import CullResult

logger = logging.getLogger(__name__)

CULLER_PREFIXES = (
    "PhotoCuller:",
    "CullerConfidence:",
    "CullerIssues:",
    "CullerSuggestedRating:",
    "CullerBurst:",
    "CullerDuplicate:",
)

# ON1 generates its own keywords ("Person", "Sky", "T-shirt") and they do pile up. An
# earlier version of this file tried to strip them using a hardcoded list of likely
# candidates, which cannot work: ON1 records its classifications as numeric ids with no
# text anywhere in the sidecar, so the list was guesswork that would happily delete a
# photographer's own "Person" or "Bridge" keyword.
#
# So preserve mode now preserves. Everything except this tool's own keywords survives.
# Use --override for a clean slate; that path also clears the classifications ON1
# regenerates its keywords from, so they do not immediately come back.


@dataclass
class WriteOptions:
    """Which parts of a result actually reach the sidecar.

    The command line writes everything, so all the flags default to on and the CLI never
    passes this object. It exists for the ON1 plugin, where the photographer sees the
    proposed metadata first and ticks the parts they want; without field-level control
    the popup's checkboxes would be decoration.

    rating is a star value rather than a flag because the plugin lets you change it
    before writing. None means "do not touch the rating", which stays the default
    everywhere -- a star rating is the photographer's own shorthand.
    """

    keywords: bool = True  # the model's descriptive keywords
    culler_keywords: bool = True  # PhotoCuller:Keep, CullerConfidence:0.82, ...
    description: bool = True
    analysis: bool = True  # the full metrics block, and the readable summary line
    rating: Optional[int] = None

    # The force flags exist for the popup. On the command line, silently keeping what
    # the photographer already wrote is the safe default. In a review window they have
    # just looked at the proposed value and ticked it, so quietly discarding it because
    # the slot was occupied would look like the button did nothing.
    force_rating: bool = False
    force_description: bool = False


def _options_for(
    result: CullResult, options: Optional[WriteOptions], suggest_ratings: bool
) -> WriteOptions:
    if options is not None:
        return options
    return WriteOptions(rating=suggested_rating(result.metrics) if suggest_ratings else None)


def culler_keywords(result: CullResult) -> List[str]:
    """The keywords the culler owns, which it also removes before each rewrite."""
    issues = ", ".join(result.issues) if result.issues else "none"
    keywords = [
        f"PhotoCuller:{result.decision}",
        f"CullerConfidence:{result.confidence:.2f}",
        f"CullerIssues:{issues}",
        f"CullerSuggestedRating:{suggested_rating(result.metrics)}",
    ]

    if result.group_size > 1:
        role = "best" if result.is_best_of_group else "alternate"
        keywords.append(f"CullerBurst:{role}-of-{result.group_size}")
    if result.duplicate_of:
        keywords.append(f"CullerDuplicate:{result.duplicate_of}")

    return keywords


def _is_culler_keyword(keyword: str) -> bool:
    return any(keyword.startswith(prefix) for prefix in CULLER_PREFIXES)


def _deduplicated(keywords: List[str]) -> List[str]:
    """Keep the first occurrence of each keyword, in order.

    Preserve mode keeps the photographer's keywords and then appends the model's, so a
    second run on the same photo used to write "bridge" twice -- and a third, three
    times. Matching is case-insensitive because photo apps treat Bridge and bridge as
    one keyword anyway.
    """
    seen = set()
    unique = []
    for keyword in keywords:
        key = keyword.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(keyword)
    return unique


def analysis_block(result: CullResult) -> Dict[str, str]:
    metrics = result.metrics
    block = {
        "decision": result.decision,
        "confidence": f"{result.confidence:.2f}",
        "issues": ", ".join(result.issues) if result.issues else "none",
        "subject_sharpness": metrics.subject_sharpness,
        "exposure": metrics.exposure,
        "framing": metrics.framing,
        "blur_score": f"{metrics.blur_score:.2f}",
        "exposure_score": f"{metrics.exposure_score:.2f}",
        "composition_score": f"{metrics.composition_score:.2f}",
        "overall_quality": f"{metrics.overall_quality:.2f}",
    }
    if metrics.verdict_reason:
        block["reason"] = metrics.verdict_reason
    if metrics.cv_sharpness is not None:
        block["measured_sharpness"] = f"{metrics.cv_sharpness:.2f}"
    return block


# ---------------------------------------------------------------------- ON1


def write_on1_sidecar(
    result: CullResult,
    override: bool = False,
    suggest_ratings: bool = False,
    options: Optional[WriteOptions] = None,
) -> bool:
    """Update the .on1 sidecar beside a photo. Returns False when there is none."""
    opts = _options_for(result, options, suggest_ratings)
    on1_file = result.filepath.with_suffix(".on1")
    if not on1_file.exists():
        return False

    try:
        data = json.loads(on1_file.read_text())
    except (OSError, ValueError) as e:
        logger.warning(f"Could not read {on1_file.name}: {e}")
        return False

    photos = data.get("photos", {})
    filename = result.filepath.name
    photo_data = next(
        (entry for entry in photos.values() if entry.get("name") == filename), None
    )
    if photo_data is None:
        return False

    metadata = photo_data.setdefault("metadata", {})
    keywords: List[str] = []

    if override:
        # ON1 regenerates its own keywords from these classifications, so leaving them
        # in place would undo the override on the next catalogue refresh.
        ml_classes = photo_data.get("ml_classes")
        if ml_classes:
            ml_classes["classifications"] = [
                c for c in ml_classes.get("classifications", [])
                if c.get("type") not in ("ONPanopticSegmenterV0", "PartialLabelingCSL")
            ]
    else:
        keywords = [kw for kw in metadata.get("Keywords", []) if not _is_culler_keyword(kw)]

    if opts.keywords and result.metrics.keywords:
        keywords.extend(result.metrics.keywords[:8])
    if opts.culler_keywords:
        keywords.extend(culler_keywords(result))

    # With both keyword sources switched off there is nothing to add, so the field is
    # left exactly as ON1 wrote it rather than being rewritten with the stripped copy.
    if opts.keywords or opts.culler_keywords or override:
        metadata["Keywords"] = _deduplicated(keywords)

    description = result.metrics.description
    if opts.description and description and (
        override or opts.force_description or not metadata.get("Description")
    ):
        metadata["Description"] = description

    # Same rule as the XMP writer: only fill an empty slot, unless told otherwise.
    if opts.rating is not None and (opts.force_rating or not metadata.get("Rating")):
        metadata["Rating"] = opts.rating

    if opts.analysis:
        metadata["PhotoCullerAnalysis"] = analysis_block(result)
    metadata["MetadataDate"] = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    metadata.setdefault("MetadataDateOffset", 0)

    try:
        on1_file.write_text(json.dumps(data, separators=(",", ":")))
        return True
    except OSError as e:
        logger.warning(f"Could not write {on1_file.name}: {e}")
        return False


# ---------------------------------------------------------------------- XMP

# The ON1 sidecar carries the full analysis as a JSON block. XMP has no equivalent
# free-form slot, so the same fields go two places: a photoculler: namespace holding
# them individually, which survives round-trips and is machine-readable, and a plain
# sentence in photoshop:Instructions, which is a field Lightroom, Bridge and ON1
# actually display. Without the second one the analysis is technically present and
# practically invisible.
XMP_NAMESPACE = "http://filipfilm.github.io/photo_culler/1.0/"

XMP_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="PhotoCuller">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"
      xmlns:photoculler="{namespace}"
      xmp:ModifyDate="{modify_date}"
      xmp:CreatorTool="PhotoCuller"{rating_attr}{analysis_attrs}>
      <dc:subject>
        <rdf:Bag>
{keywords_xml}
        </rdf:Bag>
      </dc:subject>
      <dc:title>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">{title}</rdf:li>
        </rdf:Alt>
      </dc:title>
      <dc:description>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">{description}</rdf:li>
        </rdf:Alt>
      </dc:description>
{instructions_xml}    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
"""


def _camel(key: str) -> str:
    head, *rest = key.split("_")
    return head + "".join(part.capitalize() for part in rest)


def analysis_summary(result: CullResult) -> str:
    """One readable line for the field a photo app will actually show you."""
    metrics = result.metrics
    parts = [
        f"PhotoCuller: {result.decision} ({result.confidence:.2f})",
        f"sharpness {metrics.subject_sharpness or '-'}",
        f"exposure {metrics.exposure or '-'}",
        f"framing {metrics.framing or '-'}",
    ]
    if result.group_size > 1:
        role = "best" if result.is_best_of_group else "alternate"
        parts.append(f"burst {role} of {result.group_size}")
    if result.issues:
        parts.append("issues: " + ", ".join(result.issues))
    if metrics.verdict_reason:
        parts.append(metrics.verdict_reason)
    return " | ".join(parts)


def read_existing_xmp(xmp_file: Path) -> Dict:
    existing = {"rating": None, "keywords": [], "description": ""}
    if not xmp_file.exists():
        return existing

    try:
        root = ET.parse(xmp_file).getroot()
        description_elem = next(
            (e for e in root.iter() if e.tag.endswith("Description")), None
        )
        if description_elem is None:
            return existing

        for name, value in description_elem.attrib.items():
            if name.endswith("Rating"):
                existing["rating"] = value

        for child in description_elem:
            if child.tag.endswith("subject"):
                for li in child.iter():
                    if li.tag.endswith("li") and li.text and not _is_culler_keyword(li.text):
                        existing["keywords"].append(li.text)
            elif child.tag.endswith("description"):
                for li in child.iter():
                    if li.tag.endswith("li") and li.text:
                        existing["description"] = li.text
                        break

    except (OSError, ET.ParseError) as e:
        logger.warning(f"Could not read existing XMP {xmp_file.name}: {e}")

    return existing


def xmp_path_for(photo: Path) -> Path:
    """Where this photo's XMP sidecar lives.

    Two conventions are in the wild: photo.xmp (Adobe, Lightroom, ON1) and photo.NEF.xmp
    (darktable and others). Writing the wrong one leaves a second sidecar competing with
    the catalogue's, and the photographer's existing keywords sit in a file we never
    read. So an existing sidecar always wins, and otherwise we follow Adobe.
    """
    replaced = photo.with_suffix(".xmp")
    appended = Path(str(photo) + ".xmp")

    if replaced.exists():
        return replaced
    if appended.exists():
        return appended
    return replaced


def write_xmp_sidecar(
    result: CullResult,
    override: bool = False,
    suggest_ratings: bool = False,
    options: Optional[WriteOptions] = None,
) -> bool:
    """Write a .xmp sidecar next to the photo, preserving existing user metadata."""
    opts = _options_for(result, options, suggest_ratings)
    xmp_file = xmp_path_for(result.filepath)
    existing = read_existing_xmp(xmp_file)

    keywords = [] if override else list(existing["keywords"])
    if opts.keywords and result.metrics.keywords:
        keywords.extend(result.metrics.keywords[:8])
    if opts.culler_keywords:
        keywords.extend(culler_keywords(result))

    if opts.description and result.metrics.description:
        description = result.metrics.description
    elif not override and existing["description"]:
        description = existing["description"]
    elif opts.description:
        description = f"PhotoCuller: {result.decision}"
    else:
        description = ""

    # A star rating is the photographer's own shorthand, so the culler does not get to
    # invent one unless asked. Its opinion is always available as the
    # CullerSuggestedRating keyword. An existing rating is never overwritten either way;
    # 0 means unrated, which is the only slot --suggest-ratings may fill.
    existing_rating = (existing["rating"] or "").strip()
    has_real_rating = bool(existing_rating) and existing_rating != "0"

    if has_real_rating and not opts.force_rating:
        rating_attr = f'\n      xmp:Rating="{escape(existing_rating)}"'
    elif opts.rating is not None:
        rating_attr = f'\n      xmp:Rating="{opts.rating}"'
    elif has_real_rating:
        rating_attr = f'\n      xmp:Rating="{escape(existing_rating)}"'
    else:
        rating_attr = ""

    # Descriptions and keywords are free text straight from a model, so every value has
    # to be escaped -- an unescaped ampersand produces a sidecar no photo app will open.
    keywords_xml = "\n".join(
        f"          <rdf:li>{escape(str(kw))}</rdf:li>" for kw in _deduplicated(keywords)
    )
    analysis_attrs = ""
    instructions_xml = ""
    if opts.analysis:
        analysis_attrs = "\n" + "\n".join(
            f'      photoculler:{_camel(key)}="{escape(str(value))}"'
            for key, value in analysis_block(result).items()
        )
        instructions_xml = (
            f"      <photoshop:Instructions>{escape(analysis_summary(result))}"
            "</photoshop:Instructions>\n"
        )

    content = XMP_TEMPLATE.format(
        namespace=XMP_NAMESPACE,
        modify_date=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        rating_attr=rating_attr,
        analysis_attrs=analysis_attrs,
        keywords_xml=keywords_xml,
        title=escape(result.filepath.stem),
        description=escape(description),
        instructions_xml=instructions_xml,
    )

    try:
        xmp_file.write_text(content, encoding="utf-8")
        return True
    except OSError as e:
        logger.warning(f"Could not write {xmp_file.name}: {e}")
        return False


def write_sidecar(
    result: CullResult,
    style: str,
    override: bool = False,
    suggest_ratings: bool = False,
    options: Optional[WriteOptions] = None,
) -> bool:
    if style == "on1":
        return write_on1_sidecar(result, override, suggest_ratings, options)
    if style == "xmp":
        return write_xmp_sidecar(result, override, suggest_ratings, options)
    raise ValueError(f"Unknown sidecar style: {style}")
