from rdagent.oai.backend.litellm import _redact_sensitive_settings


def test_litellm_settings_redaction_is_recursive():
    value = {
        "openai_api_key": "live-secret",
        "chat_model": "openai/glm-5.2",
        "nested": {"access_token": "nested-secret", "endpoint": "https://example.test/v1"},
    }

    redacted = _redact_sensitive_settings(value)

    assert redacted["openai_api_key"] == "***"
    assert redacted["nested"]["access_token"] == "***"
    assert redacted["chat_model"] == "openai/glm-5.2"
    assert redacted["nested"]["endpoint"] == "https://example.test/v1"
