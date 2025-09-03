"""Shared pytest fixtures for the BESO test suite."""

import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary for testing."""
    return {
        "model": {
            "name": "test_model",
            "hidden_dim": 128,
            "num_layers": 4
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10
        },
        "env": {
            "name": "block_pushing",
            "action_dim": 4,
            "obs_dim": 10
        }
    }


@pytest.fixture
def mock_observation():
    """Provide a mock observation for testing agents and environments."""
    return {
        "state": [0.1, 0.2, 0.3, 0.4, 0.5],
        "image": None,  # Could be replaced with actual image data in specific tests
        "goal": [0.8, 0.9, 1.0]
    }


@pytest.fixture
def mock_action():
    """Provide a mock action for testing."""
    return [0.5, -0.2, 0.8, 0.1]


@pytest.fixture
def mock_trajectory():
    """Provide a mock trajectory for testing data loaders."""
    return {
        "observations": [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
        "actions": [[0.5, -0.2], [0.6, -0.1], [0.7, 0.0]],
        "rewards": [0.1, 0.2, 0.5],
        "dones": [False, False, True]
    }


@pytest.fixture
def mock_device():
    """Mock device for testing (CPU)."""
    return "cpu"


@pytest.fixture
def mock_torch_model():
    """Create a mock PyTorch model for testing."""
    model = MagicMock()
    model.train.return_value = None
    model.eval.return_value = None
    model.parameters.return_value = []
    return model


@pytest.fixture
def mock_pybullet():
    """Mock pybullet for environment testing without actual physics simulation."""
    with patch('pybullet') as mock_pb:
        mock_pb.connect.return_value = 0
        mock_pb.disconnect.return_value = None
        mock_pb.loadURDF.return_value = 1
        mock_pb.resetBasePositionAndOrientation.return_value = None
        mock_pb.stepSimulation.return_value = None
        yield mock_pb


@pytest.fixture
def sample_urdf_path(temp_dir):
    """Create a sample URDF file for testing."""
    urdf_content = '''<?xml version="1.0"?>
<robot name="test_robot">
    <link name="base_link">
        <visual>
            <geometry>
                <box size="0.1 0.1 0.1"/>
            </geometry>
        </visual>
    </link>
</robot>'''
    
    urdf_path = temp_dir / "test_robot.urdf"
    urdf_path.write_text(urdf_content)
    return str(urdf_path)


@pytest.fixture
def mock_workspace_bounds():
    """Provide mock workspace bounds for testing."""
    return {
        "x_min": -0.5,
        "x_max": 0.5,
        "y_min": -0.5, 
        "y_max": 0.5,
        "z_min": 0.0,
        "z_max": 0.3
    }