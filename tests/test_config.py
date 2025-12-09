from backend.config import settings


def test_app_env_default():
    """Test that APP_ENV defaults to development."""
    assert settings.APP_ENV == "development"


def test_port_default():
    """Test that PORT defaults to 8000."""
    assert settings.PORT == 8000


def test_model_registry_path_is_non_empty():
    """Test that MODEL_REGISTRY_PATH is a non-empty string."""
    assert isinstance(settings.MODEL_REGISTRY_PATH, str)
    assert len(settings.MODEL_REGISTRY_PATH) > 0
