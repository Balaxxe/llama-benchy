from llama_benchy.client import LLMClient


def test_generation_payload_defaults():
    client = LLMClient("http://example.test/v1", "EMPTY", "model")
    messages = [{"role": "user", "content": "hello"}]

    payload = client._build_generation_payload(messages, max_tokens=128, no_cache=False)

    assert payload["model"] == "model"
    assert payload["messages"] == messages
    assert payload["max_tokens"] == 128
    assert payload["stream"] is True
    assert payload["return_token_ids"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert "min_tokens" not in payload
    assert "ignore_eos" not in payload


def test_generation_payload_merges_extra_body():
    client = LLMClient(
        "http://example.test/v1",
        "EMPTY",
        "model",
        extra_body={"temperature": 0, "ignore_eos": True},
    )

    payload = client._build_generation_payload([], max_tokens=128, no_cache=True)

    assert payload["cache_prompt"] is False
    assert payload["temperature"] == 0
    assert payload["ignore_eos"] is True


def test_exact_tg_forces_min_tokens_and_ignore_eos():
    client = LLMClient(
        "http://example.test/v1",
        "EMPTY",
        "model",
        extra_body={"max_tokens": 64, "min_tokens": 16, "ignore_eos": False},
        exact_tg=True,
    )

    payload = client._build_generation_payload([], max_tokens=128, no_cache=False)

    assert payload["max_tokens"] == 128
    assert payload["min_tokens"] == 128
    assert payload["ignore_eos"] is True
