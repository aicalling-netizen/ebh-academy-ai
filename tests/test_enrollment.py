"""Tests for the enrollment lead capture tool."""
import pytest

from core.db import normalize_phone


class TestNormalizePhone:
    def test_uae_mobile_05x(self):
        assert normalize_phone("0561234567") == "+971561234567"

    def test_uae_mobile_5x(self):
        assert normalize_phone("561234567") == "+971561234567"

    def test_with_plus_971(self):
        assert normalize_phone("+971561234567") == "+971561234567"

    def test_with_00971(self):
        assert normalize_phone("00971561234567") == "+971561234567"

    def test_with_971_no_plus(self):
        assert normalize_phone("971561234567") == "+971561234567"

    def test_with_spaces_and_dashes(self):
        assert normalize_phone("+971 56-123-4567") == "+971561234567"

    def test_international_number(self):
        result = normalize_phone("+44 7911 123456")
        assert result.startswith("+")


@pytest.mark.asyncio
async def test_enrollment_lead_missing_name():
    from tools.enrollment_lead import _handle
    result = await _handle({"phone": "0561234567"})
    assert result["status"] == "error"
    assert "name" in result["message"].lower()


@pytest.mark.asyncio
async def test_enrollment_lead_missing_phone():
    from tools.enrollment_lead import _handle
    result = await _handle({"name": "Test User"})
    assert result["status"] == "error"
    assert "phone" in result["message"].lower()


@pytest.mark.asyncio
async def test_enrollment_lead_capture_no_db():
    """Without Supabase configured, should still return captured status."""
    from tools.enrollment_lead import _handle
    result = await _handle({
        "name": "Test User",
        "phone": "0561234567",
        "email": "test@example.com",
        "course_interest": "CIDESCO Beauty Therapy",
        "notes": "Interested in installment plan",
    })
    assert result["status"] == "captured"
    assert "Test User" in result["message"]
