"""Write venv/requirements.lock from a live venv: every installed distribution at its installed version, each with
EVERY sha256 PyPI publishes for that version (as pip-compile --generate-hashes does), so `pip install --require-hashes`
installs exactly these files or fails.

    python3 venv/lock_from_venv.py <venv dir> > venv/requirements.lock

Not locked here (venv/build-venv.sh installs them from source, each its own way): boundary-lab (the pinned commit +
patches/boundary-lab/, built from a clone), podd (editable, from ~/podd), pip itself. A direct-URL distribution
(beat-engine: the fork's release wheel) is locked as its URL with the sha256 its direct_url.json recorded. A
distribution whose version PyPI does not have RAISES: there is no file to pin.
"""
import json
import subprocess
import sys
import urllib.request

NOT_LOCKED = {"boundary-lab", "podd", "pip"}


def installed(venv):
    script = ("import importlib.metadata as m, json; print(json.dumps([[d.metadata['Name'], d.version, "
              "d.read_text('direct_url.json')] for d in m.distributions()]))")
    return json.loads(subprocess.run([f"{venv}/bin/python", "-c", script], check=True, capture_output=True,
                                     text=True).stdout)


def pypi_hashes(name, version):
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as response:
        files = json.load(response)["urls"]
    if not files:
        raise SystemExit(f"(╯°□°)╯ PyPI has no files for {name}=={version}")
    return sorted(f["digests"]["sha256"] for f in files)


def main(venv):
    lines = ["# THE BLAB VENV'S LOCK: written by venv/lock_from_venv.py from the live venv; installed by",
             "# venv/build-venv.sh with --require-hashes --no-deps; checked on every run by bin/blabrun.", ""]
    for name, version, direct_url in sorted(installed(venv), key=lambda entry: entry[0].lower()):
        normalised = name.lower().replace("_", "-")
        if normalised in NOT_LOCKED:
            continue
        direct = json.loads(direct_url) if direct_url else None
        if direct and direct["url"].startswith("http"):
            sha256 = direct["archive_info"]["hash"].split("=", 1)[1]
            lines.append(f"{normalised} @ {direct['url']} \\\n    --hash=sha256:{sha256}")
            continue
        hashes = pypi_hashes(normalised, version)
        lines.append(f"{normalised}=={version} \\\n" + " \\\n".join(f"    --hash=sha256:{h}" for h in hashes))
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
