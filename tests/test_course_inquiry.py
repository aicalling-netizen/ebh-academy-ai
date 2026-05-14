"""Tests for the course inquiry tool and RAG pipeline."""
import pytest

from core.rag_pipeline import search_courses, search_faq, COURSES


class TestSearchCourses:
    def test_returns_results_for_beauty(self):
        results = search_courses("beauty therapy")
        assert len(results) >= 1
        names = [c["name"] for c in results]
        assert any("Beauty Therapy" in n for n in names)

    def test_returns_results_for_massage(self):
        results = search_courses("massage")
        assert len(results) >= 1
        names = [c["name"] for c in results]
        assert any("Madero" in n for n in names)

    def test_returns_results_for_dermaplaning(self):
        results = search_courses("dermaplaning")
        assert len(results) >= 1
        names = [c["name"] for c in results]
        assert any("Dermaplaning" in n for n in names)

    def test_returns_results_for_facial(self):
        results = search_courses("facial")
        assert len(results) >= 1
        names = [c["name"] for c in results]
        assert any("Facial" in n for n in names)

    def test_returns_results_for_management(self):
        results = search_courses("spa management")
        assert len(results) >= 1
        names = [c["name"] for c in results]
        assert any("Management" in n for n in names)

    def test_generic_query_returns_all(self):
        results = search_courses("all courses")
        assert len(results) >= 1

    def test_max_results_limit(self):
        results = search_courses("course", max_results=2)
        assert len(results) <= 2

    def test_all_courses_have_required_fields(self):
        for course in COURSES:
            assert "name" in course
            assert "duration" in course
            assert "price_aed" in course
            assert isinstance(course["price_aed"], int)
            assert "certifications" in course
            assert len(course["certifications"]) >= 1


class TestSearchFaq:
    def test_khda_query(self):
        results = search_faq("KHDA")
        assert len(results) >= 1
        assert any("KHDA" in r["q"] or "KHDA" in r["a"] for r in results)

    def test_dha_query(self):
        results = search_faq("DHA license")
        assert len(results) >= 1

    def test_payment_query(self):
        results = search_faq("payment installment")
        assert len(results) >= 1

    def test_location_query(self):
        results = search_faq("where located")
        assert len(results) >= 1

    def test_experience_query(self):
        results = search_faq("experience needed")
        assert len(results) >= 1

    def test_no_results_for_nonsense(self):
        results = search_faq("xyzzy foobar")
        assert len(results) == 0
