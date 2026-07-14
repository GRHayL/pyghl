from __future__ import annotations

import bz2
from contextlib import redirect_stderr
import io
import tempfile
import tarfile
import unittest
from pathlib import Path
from unittest import mock

from pyghl.nn_c2p import eos_catalog

CATALOG_HTML = """
<table>
  <tr><th>EOS variant</th><th>Table name and link</th></tr>
  <tr><td colspan="2"><a href="https://papers.example/ls">Lattimer and Swesty EOS</a></td></tr>
  <tr>
    <td>LS EOS, K = 220 MeV</td>
    <td><a href="EOS/LS220_table.h5.bz2">LS220_table.h5.bz2</a></td>
  </tr>
  <tr><td colspan="2"><a href="https://papers.example/hempel">Hempel et al. EOS</a></td></tr>
  <tr>
    <td>HS EOS, DD2</td>
    <td><a href="~evanoc/Hempel_DD2_table.h5.bz2">DD2</a></td>
  </tr>
  <tr><td><a href="EOS/EOSdriver.tar.gz">driver</a></td></tr>
</table>
"""

MICROPHYSICS_HTML = """
<main>
  <a href="APREOS.html">APR EOS</a>
  <a href="SROEOS.html">SRO Equation of State</a>
  <a href="equationofstate.html">O'Connor &amp; Ott EOS Tables</a>
  <a href="APREOS.html">APR EOS duplicate navigation link</a>
  <a href="nulib.html">NuLib</a>
</main>
"""

APR_CATALOG_HTML = """
<table>
  <tr><th>EOS Variant</th><th>Download link</th></tr>
  <tr><td>APR pure SNA<br>APR with NSE (3335 nuclides)</td><td>
    <a href="https://stockholmuniversity.box.com/s/pure-share">APR_0000.h5.tar.bz2</a><br>
    <a href="https://stockholmuniversity.box.com/s/nse-share">APR_3335.h5.tar.bz2</a>
  </td></tr>
</table>
"""

SRO_CATALOG_HTML = """
<table>
  <tr><th>EOS Variant</th><th>Download link</th></tr>
  <tr><td>KDE0v1 with NSE (3335 nuclides)<br>KDE0v1 pure SNA</td><td>
    <a href="EOS/KDE0v1_3335.h5.bz2">KDE0v1_3335.h5.bz2</a><br>
    <a href="EOS/KDE0v1_0000.h5.bz2">KDE0v1_0000.h5.bz2</a>
  </td></tr>
</table>
"""


class _Response(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        url: str,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(payload)
        self._url = url
        self.headers = headers or {}

    def geturl(self) -> str:
        return self._url


class _PickerScreen:
    def __init__(self, keys: list[int]):
        self.keys = iter(keys)

    def keypad(self, enabled: bool) -> None:
        pass

    def getmaxyx(self) -> tuple[int, int]:
        return 20, 120

    def erase(self) -> None:
        pass

    def border(self) -> None:
        pass

    def addstr(self, *args) -> None:
        pass

    def refresh(self) -> None:
        pass

    def getch(self) -> int:
        return next(self.keys)


class EOSCatalogTests(unittest.TestCase):
    def test_download_progress_reports_percentage_speed_and_eta(self) -> None:
        output = io.StringIO()
        clock = mock.Mock(side_effect=[0.0, 2.0])
        progress = eos_catalog._DownloadProgress(
            total_bytes=4 * 1024 * 1024,
            stream=output,
            clock=clock,
            update_interval=0.0,
        )

        progress.update(2 * 1024 * 1024)

        status = output.getvalue()
        self.assertIn("50.0%", status)
        self.assertIn("2.0 MiB/4.0 MiB", status)
        self.assertIn("1.0 MiB/s", status)
        self.assertIn("ETA 00:02", status)
        self.assertIn("[", status)
        self.assertIn("]", status)

    def test_download_progress_handles_missing_content_length(self) -> None:
        output = io.StringIO()
        clock = mock.Mock(side_effect=[0.0, 2.0])
        progress = eos_catalog._DownloadProgress(
            total_bytes=None,
            stream=output,
            clock=clock,
            update_interval=0.0,
        )

        progress.update(3 * 1024 * 1024)

        status = output.getvalue()
        self.assertIn("3.0 MiB", status)
        self.assertIn("1.5 MiB/s", status)
        self.assertIn("elapsed 00:02", status)
        self.assertNotIn("%", status)

    def test_parse_microphysics_index_discovers_all_eos_family_pages(self) -> None:
        pages = eos_catalog.parse_eos_family_pages(MICROPHYSICS_HTML)

        self.assertEqual(
            [(page.family, page.url) for page in pages],
            [
                ("APR EOS", "https://stellarcollapse.org/APREOS.html"),
                ("SRO Equation of State", "https://stellarcollapse.org/SROEOS.html"),
                (
                    "O'Connor & Ott EOS Tables",
                    "https://stellarcollapse.org/equationofstate.html",
                ),
            ],
        )

    def test_parse_catalog_builds_choices_from_hdf5_rows(self) -> None:
        tables = eos_catalog.parse_eos_catalog(CATALOG_HTML)

        self.assertEqual(
            [table.description for table in tables],
            [
                "LS EOS, K = 220 MeV",
                "HS EOS, DD2",
            ],
        )
        self.assertEqual(
            [table.family for table in tables],
            [
                "Lattimer and Swesty EOS",
                "Hempel et al. EOS",
            ],
        )
        self.assertEqual(tables[0].filename, "LS220_table.h5.bz2")
        self.assertEqual(
            tables[1].url,
            "https://stellarcollapse.org/~evanoc/Hempel_DD2_table.h5.bz2",
        )

    def test_parse_catalog_preserves_family_with_multiple_eos_references(self) -> None:
        html = """
        <tr><td colspan="2">
          <a href="nl3.html">G. Shen et al. EOS (NL3)</a>
          <a href="fsu.html">G. Shen et al. EOS (FSU)</a>
        </td></tr>
        <tr><td>GShen EOS, FSU1.7</td>
          <td><a href="~evanoc/GShenFSU.h5.bz2">table</a></td></tr>
        """

        tables = eos_catalog.parse_eos_catalog(html)

        self.assertEqual(
            tables[0].family,
            "G. Shen et al. EOS (NL3) / G. Shen et al. EOS (FSU)",
        )

    def test_parse_apr_catalog_supports_box_tar_archives(self) -> None:
        tables = eos_catalog.parse_eos_catalog(
            APR_CATALOG_HTML,
            base_url="https://stellarcollapse.org/APREOS.html",
            default_family="APR EOS",
        )

        self.assertEqual(
            [table.description for table in tables],
            [
                "APR pure SNA",
                "APR with NSE (3335 nuclides)",
            ],
        )
        self.assertEqual(
            [table.filename for table in tables],
            [
                "APR_0000.h5.tar.bz2",
                "APR_3335.h5.tar.bz2",
            ],
        )
        self.assertEqual(
            tables[0].url, "https://stockholmuniversity.box.com/s/pure-share"
        )

    def test_parse_sro_catalog_pairs_each_description_with_its_link(self) -> None:
        tables = eos_catalog.parse_eos_catalog(
            SRO_CATALOG_HTML,
            base_url="https://stellarcollapse.org/SROEOS.html",
            default_family="SRO Equation of State",
        )

        self.assertEqual(
            [table.description for table in tables],
            [
                "KDE0v1 with NSE (3335 nuclides)",
                "KDE0v1 pure SNA",
            ],
        )

    def test_parse_catalog_rejects_downloads_from_other_hosts(self) -> None:
        html = '<tr><td>Bad</td><td><a href="https://example.com/bad.h5.bz2">bad</a></td></tr>'

        with self.assertRaisesRegex(ValueError, "stellarcollapse.org"):
            eos_catalog.parse_eos_catalog(html)

    def test_fetch_catalog_discovers_and_combines_all_family_pages(self) -> None:
        pages = {
            eos_catalog.MICROPHYSICS_URL: MICROPHYSICS_HTML,
            "https://stellarcollapse.org/APREOS.html": APR_CATALOG_HTML,
            "https://stellarcollapse.org/SROEOS.html": SRO_CATALOG_HTML,
            "https://stellarcollapse.org/equationofstate.html": CATALOG_HTML,
        }

        def open_url(url: str, *, timeout: float):
            self.assertEqual(timeout, eos_catalog.NETWORK_TIMEOUT_SECONDS)
            return _Response(pages[url].encode(), url)

        tables = eos_catalog.fetch_eos_catalog(open_url=open_url)

        self.assertEqual(len(tables), 6)
        self.assertEqual(
            {table.family for table in tables},
            {
                "APR EOS",
                "SRO Equation of State",
                "Lattimer and Swesty EOS",
                "Hempel et al. EOS",
            },
        )

    def test_fetch_catalog_recursively_discovers_nested_eos_pages(self) -> None:
        index_html = '<a href="parent.html">Parent EOS</a>'
        parent_html = '<a href="nested-eos">Additional EOS tables</a>'
        nested_html = """
        <table><tr><td>Nested variant</td><td>
          <a href="EOS/nested.h5.bz2">nested.h5.bz2</a>
        </td></tr></table>
        """
        pages = {
            eos_catalog.MICROPHYSICS_URL: index_html,
            "https://stellarcollapse.org/parent.html": parent_html,
            "https://stellarcollapse.org/nested-eos": nested_html,
        }

        def open_url(url: str, *, timeout: float):
            return _Response(pages[url].encode(), url)

        tables = eos_catalog.fetch_eos_catalog(open_url=open_url)

        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].family, "Additional EOS tables")

    def test_fetch_catalog_rejects_oversized_html(self) -> None:
        payload = b"x" * (eos_catalog.MAX_CATALOG_BYTES + 1)
        opener = mock.Mock(
            return_value=_Response(payload, eos_catalog.MICROPHYSICS_URL)
        )

        with self.assertRaisesRegex(OSError, "catalog exceeds"):
            eos_catalog.fetch_eos_catalog(open_url=opener)

    def test_download_decompresses_selected_table_atomically(self) -> None:
        table = eos_catalog.EOSTable(
            family="Lattimer and Swesty EOS",
            description="LS EOS",
            filename="LS220_table.h5.bz2",
            url="https://stellarcollapse.org/EOS/LS220_table.h5.bz2",
        )
        payload = b"an hdf5 payload"
        opener = mock.Mock(return_value=_Response(bz2.compress(payload), table.url))

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = eos_catalog.download_eos_table(
                table,
                destination_dir=Path(temp_dir),
                open_url=opener,
            )

            self.assertEqual(destination, Path(temp_dir) / "LS220_table.h5")
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse((Path(temp_dir) / "LS220_table.h5.bz2.part").exists())
            self.assertFalse((Path(temp_dir) / "LS220_table.h5.part").exists())

    def test_download_uses_response_size_for_progress(self) -> None:
        table = eos_catalog.EOSTable(
            family="Lattimer and Swesty EOS",
            description="LS EOS",
            filename="LS220_table.h5.bz2",
            url="https://stellarcollapse.org/EOS/LS220_table.h5.bz2",
        )
        compressed = bz2.compress(b"an hdf5 payload")
        opener = mock.Mock(
            return_value=_Response(
                compressed,
                table.url,
                headers={"Content-Length": str(len(compressed))},
            )
        )
        progress_output = io.StringIO()

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            redirect_stderr(progress_output),
        ):
            eos_catalog.download_eos_table(
                table,
                destination_dir=Path(temp_dir),
                open_url=opener,
            )

        self.assertIn("100.0%", progress_output.getvalue())

    def test_download_extracts_hdf5_from_tar_bz2_archive(self) -> None:
        table = eos_catalog.EOSTable(
            family="APR EOS",
            description="APR pure SNA",
            filename="APR_0000.h5.tar.bz2",
            url="https://stockholmuniversity.box.com/s/pure-share",
        )
        payload = b"APR hdf5 payload"
        archive_buffer = io.BytesIO()
        with tarfile.open(fileobj=archive_buffer, mode="w:bz2") as archive:
            member = tarfile.TarInfo("tables/APR_0000.h5")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        opener = mock.Mock(return_value=_Response(archive_buffer.getvalue(), table.url))

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = eos_catalog.download_eos_table(
                table,
                destination_dir=Path(temp_dir),
                open_url=opener,
            )

            self.assertEqual(destination, Path(temp_dir) / "APR_0000.h5")
            self.assertEqual(destination.read_bytes(), payload)

    def test_download_uses_existing_decompressed_table(self) -> None:
        table = eos_catalog.EOSTable(
            family="Lattimer and Swesty EOS",
            description="LS EOS",
            filename="LS220_table.h5.bz2",
            url="https://stellarcollapse.org/EOS/LS220_table.h5.bz2",
        )
        opener = mock.Mock(side_effect=AssertionError("network should not be used"))

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "LS220_table.h5"
            destination.write_bytes(b"cached")

            result = eos_catalog.download_eos_table(
                table,
                destination_dir=Path(temp_dir),
                open_url=opener,
            )

            self.assertEqual(result, destination)
            opener.assert_not_called()

    def test_download_rejects_directory_at_destination_path(self) -> None:
        table = eos_catalog.EOSTable(
            family="Lattimer and Swesty EOS",
            description="LS EOS",
            filename="LS220_table.h5.bz2",
            url="https://stellarcollapse.org/EOS/LS220_table.h5.bz2",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "LS220_table.h5"
            destination.mkdir()

            with self.assertRaises(FileExistsError):
                eos_catalog.download_eos_table(table, destination_dir=Path(temp_dir))

    def test_download_cleans_partial_files_after_invalid_archive(self) -> None:
        table = eos_catalog.EOSTable(
            family="Lattimer and Swesty EOS",
            description="LS EOS",
            filename="LS220_table.h5.bz2",
            url="https://stellarcollapse.org/EOS/LS220_table.h5.bz2",
        )
        opener = mock.Mock(return_value=_Response(b"not bzip2", table.url))

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(OSError):
                eos_catalog.download_eos_table(
                    table,
                    destination_dir=Path(temp_dir),
                    open_url=opener,
                )

            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    def test_choice_filter_matches_description_and_filename(self) -> None:
        table = eos_catalog.EOSTable(
            family="Hempel et al. EOS",
            description="HS EOS, DD2",
            filename="Hempel_DD2_table.h5.bz2",
            url="https://stellarcollapse.org/~evanoc/Hempel_DD2_table.h5.bz2",
        )

        self.assertTrue(eos_catalog.choice_matches(table, "hs dd2"))
        self.assertTrue(eos_catalog.choice_matches(table, "hempel table"))
        self.assertFalse(eos_catalog.choice_matches(table, "sfho"))

    def test_picker_treats_lowercase_k_as_filter_text(self) -> None:
        tables = [
            eos_catalog.EOSTable(
                family="Other EOS",
                description="S variant",
                filename="S.h5.bz2",
                url="https://stellarcollapse.org/EOS/S.h5.bz2",
            ),
            eos_catalog.EOSTable(
                family="SRO EOS",
                description="SkT1 variant",
                filename="SkT1.h5.bz2",
                url="https://stellarcollapse.org/EOS/SkT1.h5.bz2",
            ),
        ]
        screen = _PickerScreen([ord("S"), ord("k"), 10])

        selected = eos_catalog._run_eos_picker(screen, tables)

        self.assertEqual(selected.description, "SkT1 variant")

    def test_choose_and_download_fetches_selects_then_downloads(self) -> None:
        table = eos_catalog.EOSTable(
            family="Hempel et al. EOS",
            description="HS EOS, DD2",
            filename="DD2.h5.bz2",
            url="https://stellarcollapse.org/~evanoc/DD2.h5.bz2",
        )
        fetch = mock.Mock(return_value=[table])
        select = mock.Mock(return_value=table)
        download = mock.Mock(return_value=Path("DD2.h5"))

        result = eos_catalog.choose_and_download_eos(
            fetch_catalog=fetch,
            select_table=select,
            download_table=download,
        )

        self.assertEqual(result, Path("DD2.h5"))
        select.assert_called_once_with([table])
        download.assert_called_once_with(table, destination_dir=Path.cwd())


if __name__ == "__main__":
    unittest.main()
