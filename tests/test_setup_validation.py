"""Validation tests to ensure the testing infrastructure is properly set up."""

import pytest
from pathlib import Path


def test_pytest_working():
    """Basic test to verify pytest is working."""
    assert True


def test_pytest_markers():
    """Test that custom markers are properly configured."""
    # This test should pass and verify our markers work
    pass


@pytest.mark.unit
def test_unit_marker():
    """Test that the unit marker works."""
    assert 1 + 1 == 2


@pytest.mark.integration
def test_integration_marker():
    """Test that the integration marker works."""
    assert "test" in "testing"


@pytest.mark.slow
def test_slow_marker():
    """Test that the slow marker works."""
    import time
    time.sleep(0.01)  # Small sleep to simulate slow test
    assert True


def test_fixtures_available(temp_dir, mock_config, mock_observation):
    """Test that fixtures from conftest.py are available."""
    # Test temp_dir fixture
    assert temp_dir.exists()
    assert temp_dir.is_dir()
    
    # Test mock_config fixture
    assert isinstance(mock_config, dict)
    assert "model" in mock_config
    assert "training" in mock_config
    
    # Test mock_observation fixture
    assert isinstance(mock_observation, dict)
    assert "state" in mock_observation


def test_beso_package_importable():
    """Test that the beso package can be imported."""
    try:
        import beso
        assert hasattr(beso, '__init__')
    except ImportError:
        pytest.skip("BESO package dependencies not installed yet")


class TestProjectStructure:
    """Tests to validate the project structure."""
    
    def test_project_root_exists(self):
        """Test that expected project files exist."""
        project_root = Path(__file__).parent.parent
        
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / "beso").exists()
        assert (project_root / "beso" / "__init__.py").exists()
    
    def test_test_structure_exists(self):
        """Test that test directory structure is correct."""
        test_dir = Path(__file__).parent
        
        assert (test_dir / "__init__.py").exists()
        assert (test_dir / "conftest.py").exists()
        assert (test_dir / "unit").exists()
        assert (test_dir / "integration").exists()
        assert (test_dir / "unit" / "__init__.py").exists()
        assert (test_dir / "integration" / "__init__.py").exists()