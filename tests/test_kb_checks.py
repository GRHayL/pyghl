from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPOSITORY_ROOT / "scripts" / "check_kb.py"


class KnowledgeBaseCheckerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        self.root.mkdir()

    def write(self, relative: str, text: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def valid_root(self, index_extra: str = "", root_extra: str = "") -> None:
        self.write(
            "AGENTS.md",
            "# Fixture Knowledge Base\n\n"
            "This root routes fixture knowledge.\n\n"
            "[Index](wiki/index.md)\n"
            f"{root_extra}",
        )
        self.write(
            "wiki/index.md",
            "# Fixture Index\n\n"
            "This page routes fixture details.\n"
            f"{index_extra}",
        )

    def run_checker(self) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [sys.executable, str(CHECKER), "--root", str(self.root)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

    def assert_passes(self) -> subprocess.CompletedProcess[str]:
        result = self.run_checker()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("KB check: 0 error(s)", result.stdout)
        return result

    def assert_fails_with(self, code: str) -> subprocess.CompletedProcess[str]:
        result = self.run_checker()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f" {code} ", result.stdout)
        return result

    def git(self, root: Path, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["GIT_OPTIONAL_LOCKS"] = "0"
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def init_git(self, root: Path) -> None:
        self.git(root, "init", "-q")
        self.git(root, "config", "user.name", "Fixture")
        self.git(root, "config", "user.email", "fixture@example.invalid")

    def add_parent_gitlink(self, pin: str) -> None:
        self.init_git(self.root)
        self.git(self.root, "add", "AGENTS.md", "wiki")
        self.git(
            self.root,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{pin},extern/GRHayL",
        )
        self.git(self.root, "commit", "-qm", "fixture parent")

    @staticmethod
    def snapshot(root: Path) -> list[tuple[str, str, int, str]]:
        rows: list[tuple[str, str, int, str]] = []
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            stat_result = path.lstat()
            if path.is_symlink():
                rows.append((relative, "symlink", stat_result.st_mode, os.readlink(path)))
            elif path.is_file():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                rows.append((relative, "file", stat_result.st_mode, digest))
            elif path.is_dir():
                rows.append((relative, "dir", stat_result.st_mode, ""))
        return rows

    def test_valid_graph_reference_fragment_and_fences(self) -> None:
        self.valid_root(
            "\n[Details][detail-ref]\n\n[detail-ref]: detail.md#proof-details\n"
        )
        self.write(
            "wiki/detail.md",
            "# Fixture Detail\n\n"
            "This page owns fixture detail.\n\n"
            "## Proof Details\n\n"
            "```markdown\n[ignored](missing.md)\n# ignored H1\n```\n\n"
            "Inline code `[also ignored](missing.md)` is manual.\n",
        )
        self.assert_passes()

    def test_broken_escaping_absolute_and_file_uri_links(self) -> None:
        cases = {
            "LINK_EMPTY": "\n[Bad]()\n",
            "LINK_MISSING": "\n[Bad](missing.md)\n",
            "LINK_ESCAPE": "\n[Bad](../../outside.md)\n",
            "LINK_ABSOLUTE": "\n[Bad](/work/private.md)\n",
            "FILE_URI": "\n[Bad](file:///work/private.md)\n",
            "WORKSPACE_ABSOLUTE": f"\nStored workspace path: `{self.root}/private.md`.\n",
        }
        for code, extra in cases.items():
            with self.subTest(code=code):
                shutil.rmtree(self.root)
                self.root.mkdir()
                self.valid_root(extra)
                self.assert_fails_with(code)

    def test_symlink_escape_fails(self) -> None:
        outside = Path(self.temporary.name) / "outside.md"
        outside.write_text("outside\n", encoding="utf-8")
        self.valid_root("\n[Bad](outside.md)\n")
        (self.root / "wiki" / "outside.md").symlink_to(outside)
        self.assert_fails_with("LINK_ESCAPE")

    def test_checked_page_symlink_escape_fails(self) -> None:
        outside = Path(self.temporary.name) / "outside-page.md"
        outside.write_text("# Outside\n\nThis page lives outside repository.\n", encoding="utf-8")
        self.valid_root("\n[Bad page](outside-page.md)\n")
        (self.root / "wiki" / "outside-page.md").symlink_to(outside)
        self.assert_fails_with("PAGE_ESCAPE")

    def test_malformed_url_is_diagnostic_not_crash(self) -> None:
        self.valid_root("\n[Malformed](https://[invalid)\n")
        self.assert_fails_with("LINK_INVALID")

    def test_case_mismatch_and_case_collisions_fail(self) -> None:
        self.valid_root("\n[Detail](Detail.md)\n")
        self.write(
            "wiki/detail.md",
            "# Detail\n\nThis page owns lowercase detail.\n",
        )
        self.assert_fails_with("LINK_CASE")

        shutil.rmtree(self.root)
        self.root.mkdir()
        self.valid_root("\n[Upper](Case.md)\n[Lower](case.md)\n")
        self.write("wiki/Case.md", "# Upper\n\nThis page is uppercase.\n")
        self.write("wiki/case.md", "# Lower\n\nThis page is lowercase.\n")
        self.assert_fails_with("PAGE_CASE_COLLISION")

    def test_missing_fragment_fails(self) -> None:
        self.valid_root("\n[Missing fragment](#not-present)\n")
        self.assert_fails_with("FRAGMENT_MISSING")

    def test_orphan_fails(self) -> None:
        self.valid_root()
        self.write("wiki/orphan.md", "# Orphan\n\nThis page has no inbound route.\n")
        self.assert_fails_with("ORPHAN")

    def test_h1_and_early_purpose_rules(self) -> None:
        cases = {
            "missing": ("No heading here.\n", "H1_COUNT"),
            "duplicate": (
                "# First\n\nThis page starts correctly.\n\n# Second\n",
                "H1_COUNT",
            ),
            "purpose": ("# Heading\n\n## Details\n\nText arrives too late.\n", "EARLY_PURPOSE"),
        }
        for name, (content, code) in cases.items():
            with self.subTest(name=name):
                shutil.rmtree(self.root)
                self.root.mkdir()
                self.valid_root("\n[Detail](detail.md)\n")
                self.write("wiki/detail.md", content)
                self.assert_fails_with(code)

    def test_undefined_reference_fails(self) -> None:
        self.valid_root("\n[Missing][not-defined]\n")
        self.assert_fails_with("REFERENCE_UNDEFINED")

    def test_allowed_policy_hashes_timestamps_and_provenance(self) -> None:
        self.valid_root(
            "\nDo not store `last_verified: 2026-01-01`, `source_digest = abc`, "
            "`source_checksum: abc`, `source_mtime = 1`, or `kb_fingerprint: x`.\n\n"
            "Product fields remain allowed: canonical MD5 `d41d8cd98f00b204e9800998ecf8427e`, "
            "file SHA-1 `da39a3ee5e6b4b0d3255bfef95601890afd80709`, "
            "`perm_sha1_16`, C header guard SHA-1, artifact timestamp, and provenance.\n"
            "canonical_md5 = d41d8cd98f00b204e9800998ecf8427e; "
            "file_md5: d41d8cd98f00b204e9800998ecf8427e; perm_sha1_16 = abc; "
            "header_guard_sha1: abc; artifact_timestamp = 2026-01-01; "
            "provenance: exact-artifact.\n"
        )
        self.assert_passes()

    def test_affirmative_freshness_assignments_fail(self) -> None:
        fields = (
            "last_verified",
            "source_digest",
            "source_checksum",
            "source_mtime",
            "kb_fingerprint",
        )
        for field in fields:
            for operator in (":", "="):
                with self.subTest(field=field, operator=operator):
                    shutil.rmtree(self.root)
                    self.root.mkdir()
                    self.valid_root(f"\n{field} {operator} stored-value\n")
                    self.assert_fails_with("FRESHNESS_FIELD")

        shutil.rmtree(self.root)
        self.root.mkdir()
        self.valid_root("\nStored metadata: `last_verified: 2026-01-01`.\n")
        self.assert_fails_with("FRESHNESS_FIELD")

    def test_missing_nested_repository_skips_pin_proof(self) -> None:
        self.valid_root(root_extra="\n[Pinned](extern/GRHayL/AGENTS.md)\n")
        pin = "1" * 40
        self.add_parent_gitlink(pin)
        result = self.assert_passes()
        self.assertIn(" PIN_PROOF_SKIPPED nested repository unavailable", result.stdout)
        self.assertIn("1 proof skip(s)", result.stdout)

    def test_missing_pin_object_skips_pin_proof(self) -> None:
        self.valid_root(root_extra="\n[Pinned](extern/GRHayL/AGENTS.md)\n")
        nested = self.root / "extern" / "GRHayL"
        nested.mkdir(parents=True)
        self.init_git(nested)
        pin = "2" * 40
        self.add_parent_gitlink(pin)
        result = self.assert_passes()
        self.assertIn(" PIN_PROOF_SKIPPED parent pin object", result.stdout)
        self.assertIn("1 proof skip(s)", result.stdout)

    def test_advanced_checkout_cannot_mask_missing_pin_target(self) -> None:
        self.valid_root(root_extra="\n[Pinned](extern/GRHayL/only-advanced.md)\n")
        nested = self.root / "extern" / "GRHayL"
        nested.mkdir(parents=True)
        self.init_git(nested)
        (nested / "README.md").write_text("base\n", encoding="utf-8")
        self.git(nested, "add", "README.md")
        self.git(nested, "commit", "-qm", "base")
        pin = self.git(nested, "rev-parse", "HEAD").stdout.strip()
        self.add_parent_gitlink(pin)

        (nested / "only-advanced.md").write_text("advanced\n", encoding="utf-8")
        self.git(nested, "add", "only-advanced.md")
        self.git(nested, "commit", "-qm", "advanced")
        result = self.assert_fails_with("PIN_TARGET_MISSING")
        self.assertIn(f"target absent at parent pin {pin}", result.stdout)

    def test_checker_is_read_only_and_diagnostics_are_stable(self) -> None:
        self.valid_root("\n[Second](z-missing.md)\n[First](a-missing.md)\n")
        self.init_git(self.root)
        self.git(self.root, "add", "AGENTS.md", "wiki/index.md")
        self.git(self.root, "commit", "-qm", "fixture")
        self.write("untracked.txt", "preserve me\n")
        checker_before = CHECKER.read_bytes()
        before = self.snapshot(self.root)
        result = self.assert_fails_with("LINK_MISSING")
        after = self.snapshot(self.root)
        self.assertEqual(before, after)
        self.assertEqual(checker_before, CHECKER.read_bytes())
        diagnostics = [line for line in result.stdout.splitlines() if line.startswith("ERROR ")]
        self.assertEqual(diagnostics, sorted(diagnostics))


if __name__ == "__main__":
    unittest.main()
