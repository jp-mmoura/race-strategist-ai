"""
Unit tests for query parsing and FastF1 tool behaviour.

Run with:
    pytest tests/test_tools.py -v
"""
from __future__ import annotations

import socket

import pytest
from unittest.mock import MagicMock, patch
from requests.exceptions import Timeout

from graph.nodes import _parse_query


# ===================================================================
# 1. _parse_query — free text → (circuit, year, session_type)
# ===================================================================

class TestParseQuery:
    """Tests cover circuit names, aliases, cities, countries, and PT/EN queries."""

    # -- English ----------------------------------------------------------

    def test_en_circuit_name(self):
        circuit, year, session = _parse_query("Monaco 2022 race")
        assert circuit == "Monaco"
        assert year == 2022
        assert session == "R"

    def test_en_city_alias_monza(self):
        circuit, year, session = _parse_query("Monza 2021 qualifying")
        assert circuit == "Monza"
        assert year == 2021
        assert session == "Q"

    def test_en_country_alias_italy(self):
        circuit, year, session = _parse_query("Italy 2023 race")
        assert circuit == "Monza"
        assert year == 2023

    def test_en_multi_word_abu_dhabi(self):
        circuit, year, session = _parse_query("Abu Dhabi 2023 race")
        assert circuit == "Abu Dhabi"
        assert year == 2023

    def test_en_multi_word_saudi_arabia(self):
        circuit, year, session = _parse_query("Saudi Arabia 2023 race")
        assert circuit == "Jeddah"
        assert year == 2023

    def test_en_abbreviation_cota(self):
        circuit, year, session = _parse_query("COTA 2022 race")
        assert circuit == "Austin"
        assert year == 2022

    def test_en_sprint_session(self):
        circuit, year, session = _parse_query("Spa 2023 sprint")
        assert circuit == "Spa"
        assert session == "S"

    def test_en_fp1_session(self):
        circuit, year, session = _parse_query("Barcelona FP1 2023")
        assert circuit == "Barcelona"
        assert session == "FP1"

    def test_en_no_year(self):
        circuit, year, session = _parse_query("Silverstone race strategy")
        assert circuit == "Silverstone"
        assert year is None

    def test_en_no_circuit(self):
        circuit, year, session = _parse_query("2023 race strategy")
        assert circuit is None
        assert year == 2023

    # -- Portuguese -------------------------------------------------------

    def test_pt_brasil(self):
        circuit, year, session = _parse_query("brasil 2022 corrida")
        assert circuit == "São Paulo"
        assert year == 2022
        assert session == "R"

    def test_pt_italia_accented(self):
        circuit, year, session = _parse_query("Estratégia para a corrida da itália 2023")
        assert circuit == "Monza"
        assert year == 2023

    def test_pt_belgica_accented(self):
        circuit, year, session = _parse_query("bélgica 2021 classificação")
        assert circuit == "Spa"
        assert year == 2021
        assert session == "Q"

    def test_pt_austria_accented(self):
        circuit, year, session = _parse_query("áustria 2023 corrida")
        assert circuit == "Spielberg"
        assert year == 2023

    def test_pt_monaco_accented(self):
        circuit, year, session = _parse_query("estratégia para o mônaco 2024")
        assert circuit == "Monaco"
        assert year == 2024

    # -- Other aliases ----------------------------------------------------

    def test_alias_interlagos(self):
        circuit, year, session = _parse_query("Interlagos 2019 race")
        assert circuit == "São Paulo"
        assert year == 2019

    def test_alias_sakhir(self):
        circuit, year, session = _parse_query("sakhir 2020 race")
        assert circuit == "Bahrain"

    def test_empty_string(self):
        circuit, year, session = _parse_query("")
        assert circuit is None
        assert year is None
        assert session is None


# ===================================================================
# 2. Year extraction — regex boundary behaviour
# ===================================================================

class TestYearValidation:
    """
    _parse_query uses the pattern r"\\b(201[8-9]|202[0-9])\\b": 2018–2029 only.
    Any year outside that window — including 2000–2017 — is treated as
    absent (year=None).  FastF1 has reliable data only from 2018 onward;
    out-of-range detection is the caller's responsibility (supervisor_node).
    """

    @pytest.mark.parametrize("query", [
        "Monaco 1998 race",
        "Monza 1999 race",
        "Silverstone 1985 race",
    ])
    def test_pre_2000_returns_none(self, query: str):
        _, year, _ = _parse_query(query)
        assert year is None, f"Expected None for pre-2000 year in: {query!r}"

    @pytest.mark.parametrize("query", [
        "Monaco 2030 race",
        "Monza 2035 race",
        "Silverstone 2099 race",
    ])
    def test_post_2029_returns_none(self, query: str):
        _, year, _ = _parse_query(query)
        assert year is None, f"Expected None for post-2029 year in: {query!r}"

    @pytest.mark.parametrize("year_val", [2018, 2019, 2021, 2023, 2024, 2029])
    def test_modern_years_are_extracted(self, year_val: int):
        _, year, _ = _parse_query(f"Monaco {year_val} race")
        assert year == year_val

    @pytest.mark.parametrize("year_val", [2000, 2010, 2015, 2017])
    def test_pre_2018_years_return_none(self, year_val: int):
        # Regex now requires 2018–2029; pre-2018 years are excluded because
        # FastF1 has no reliable session data before the 2018 season.
        _, year, _ = _parse_query(f"Monaco {year_val} race")
        assert year is None, f"Expected None for pre-2018 year {year_val}"

    def test_no_year_returns_none(self):
        _, year, _ = _parse_query("Monaco race strategy")
        assert year is None


# ===================================================================
# 3. FastF1 tool — timeout propagation
# ===================================================================

class TestFastF1ToolTimeout:
    """
    tools/fastf1_tool.get_session() calls fastf1.get_session() then
    session.load() with no built-in timeout.  Network exceptions must
    propagate to the caller rather than hang indefinitely.
    FastF1 is fully mocked so no network or filesystem access occurs.
    """

    def test_timeout_raised_by_session_load(self):
        """session.load() raising Timeout must propagate unchanged."""
        from tools.fastf1_tool import get_session

        mock_session = MagicMock()
        mock_session.load.side_effect = Timeout("Connection timed out")

        with patch("tools.fastf1_tool.fastf1.get_session", return_value=mock_session):
            with pytest.raises(Timeout):
                get_session(2023, "Monaco", "R")

    def test_timeout_raised_by_get_session(self):
        """fastf1.get_session() raising Timeout must propagate unchanged."""
        from tools.fastf1_tool import get_session

        with patch("tools.fastf1_tool.fastf1.get_session", side_effect=Timeout("DNS timed out")):
            with pytest.raises(Timeout):
                get_session(2023, "Monaco", "R")

    def test_socket_timeout_propagates(self):
        """socket.timeout (urllib3 layer) must also propagate."""
        from tools.fastf1_tool import get_session

        mock_session = MagicMock()
        mock_session.load.side_effect = socket.timeout("timed out")

        with patch("tools.fastf1_tool.fastf1.get_session", return_value=mock_session):
            with pytest.raises(socket.timeout):
                get_session(2023, "Monza", "Q")

    def test_successful_call_returns_session(self):
        """Smoke test: a cooperative mock returns the session object."""
        from tools.fastf1_tool import get_session

        mock_session = MagicMock()
        mock_session.load.return_value = None

        with patch("tools.fastf1_tool.fastf1.get_session", return_value=mock_session):
            result = get_session(2023, "Monaco", "R")

        assert result is mock_session
        mock_session.load.assert_called_once()
