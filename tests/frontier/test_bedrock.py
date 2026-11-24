"""Tests for qlora_finetuning.frontier.bedrock, with a stub client. No call reaches AWS."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from botocore.exceptions import (
    ClientError,
    CredentialRetrievalError,
    EndpointConnectionError,
    NoCredentialsError,
    ProfileNotFound,
    ReadTimeoutError,
    SSOTokenLoadError,
    TokenRetrievalError,
    UnauthorizedSSOTokenError,
)

from qlora_finetuning import aws_common
from qlora_finetuning.evaluation.predictor import PredictorError
from qlora_finetuning.frontier.bedrock import (
    BedrockProvider,
    aws_session,
    bedrock_runtime_client,
    build_request,
    check_identity,
    client_config,
    read_response,
)
from qlora_finetuning.frontier.config import load_frontier_config
from qlora_finetuning.frontier.provider import FrontierProvider, ProviderReply
from qlora_finetuning.settings import ConfigError

MODEL = "amazon.nova-lite-v1:0"
# Invented texts with what a wrapper could damage: line breaks, a trailing space, a tab,
# characters outside ASCII, and braces.
SYSTEM = "You are a zebrafish accountant.\n\nReply with two lines: £ and € count. "
USER = "Context:\n- (none)\n\nTable:\n| item | 2031 |\n|---|---|\n| kelp\t| 920 |\n\nWhy {x}? "
AWS = {"profile": "qlora", "account_id": "123456789012", "region": "us-east-1"}

AUTH_ERRORS = [
    UnauthorizedSSOTokenError(),
    SSOTokenLoadError(error_msg="stubbed"),
    TokenRetrievalError(provider="sso", error_msg="stubbed"),
    NoCredentialsError(),
    CredentialRetrievalError(provider="sso", error_msg="stubbed"),
    ProfileNotFound(profile="qlora"),
]
ERROR_CODES = [
    "ThrottlingException",
    "ModelNotReadyException",
    "ModelTimeoutException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelErrorException",
    "ValidationException",
    "AccessDeniedException",
    "ResourceNotFoundException",
]
EXPIRED_CODES = [
    "ExpiredToken",
    "ExpiredTokenException",
    "RequestExpired",
    "InvalidClientTokenId",
    "UnrecognizedClientException",
]


@pytest.fixture
def cfg():
    return load_frontier_config()


def client_error(code: str, message: str = "stubbed", operation: str = "Converse") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}}, operation)


def response(text="PROGRAM: add(1, 2)\nANSWER: 3", stop="end_turn", **changes) -> dict:
    data = {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": stop,
        "usage": {"inputTokens": 640, "outputTokens": 17, "totalTokens": 657},
        "metrics": {"latencyMs": 812},
    }
    data.update(changes)
    return data


def provider_with(cfg, result=None, error=None, clock=None):
    client = MagicMock()
    if error is not None:
        client.converse.side_effect = error
    else:
        client.converse.return_value = response() if result is None else result
    ticks = iter(clock or (10.0, 11.5))
    return BedrockProvider(client, cfg, profile="qlora", clock=lambda: next(ticks)), client


# --------------------------------------------------------------------------- the request


def test_the_request_holds_both_texts_character_for_character(cfg):
    request = build_request(MODEL, SYSTEM, USER, 192, cfg)
    assert request["system"] == [{"text": SYSTEM}]
    assert request["messages"] == [{"role": "user", "content": [{"text": USER}]}]
    assert request["system"][0]["text"] is SYSTEM and len(request["messages"]) == 1


def test_the_request_sends_greedy_decoding_and_nothing_else(cfg):
    request = build_request(MODEL, SYSTEM, USER, 192, cfg)
    assert set(request) == {
        "modelId",
        "system",
        "messages",
        "inferenceConfig",
        "additionalModelRequestFields",
    }
    assert request["modelId"] == MODEL
    assert request["inferenceConfig"] == {"maxTokens": 192, "temperature": 0}
    assert request["additionalModelRequestFields"] == {"inferenceConfig": {"topK": 1}}
    flat = repr(request)
    for unsent in ("topP", "stopSequences", "cachePoint", "reasoningConfig", "serviceTier",
                   "performanceConfig", "guardrailConfig", "toolConfig"):
        assert unsent not in flat, unsent


def test_the_provider_sends_exactly_that_request_once(cfg):
    provider, client = provider_with(cfg)
    reply = provider.converse(MODEL, SYSTEM, USER, 192)
    client.converse.assert_called_once_with(**build_request(MODEL, SYSTEM, USER, 192, cfg))
    assert len(client.mock_calls) == 1  # nothing else was asked of the client
    assert reply == ProviderReply(
        text="PROGRAM: add(1, 2)\nANSWER: 3",
        input_tokens=640,
        output_tokens=17,
        stop_reason="end",
        latency_seconds=1.5,  # our own clock around the call, not the service's 0.812
    )
    assert isinstance(provider, FrontierProvider)


# --------------------------------------------------------------------------- the response


def test_text_blocks_are_joined_and_other_blocks_left_out():
    content = [
        {"reasoningContent": {"reasoningText": {"text": "thinking about kelp"}}},
        {"text": "PROGRAM: add(1, 2)\n"},
        {"toolUse": {"toolUseId": "t", "name": "n", "input": {}}},
        {"text": "ANSWER: 3 "},
    ]
    reply = read_response(
        response(output={"message": {"role": "assistant", "content": content}}), 0.5
    )
    assert reply.text == "PROGRAM: add(1, 2)\nANSWER: 3 "  # unchanged, trailing space kept
    assert "kelp" not in reply.text


@pytest.mark.parametrize(
    ("stop", "expected"),
    [
        ("end_turn", "end"),
        ("stop_sequence", "end"),
        ("max_tokens", "length"),
        ("tool_use", "other"),
        ("guardrail_intervened", "other"),
        ("content_filtered", "other"),
        ("malformed_model_output", "other"),
        ("malformed_tool_use", "other"),
        ("model_context_window_exceeded", "other"),
        ("a_reason_added_later", "other"),
    ],
)
def test_each_stop_reason_maps_to_one_of_the_runners_three(stop, expected):
    assert read_response(response(stop=stop), 0.1).stop_reason == expected


@pytest.mark.parametrize("content", [[], [{"text": ""}], [{"image": {}}]])
def test_an_empty_reply_is_a_reply_not_a_failure(content):
    reply = read_response(
        response(stop="content_filtered", output={"message": {"role": "assistant",
                                                              "content": content}}),
        0.1,
    )
    assert reply.text == "" and reply.stop_reason == "other" and reply.input_tokens == 640


def test_zero_tokens_are_whole_numbers():
    usage = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
    assert read_response(response(usage=usage), 0.1).output_tokens == 0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"output": {}}, "no assistant message"),
        ({"output": None}, "no assistant message"),
        ({"output": {"message": {"role": "user", "content": []}}}, "no assistant message"),
        ({"output": {"message": {"role": "assistant"}}}, "no content list"),
        ({"output": {"message": {"role": "assistant", "content": "text"}}}, "no content list"),
        ({"stopReason": None}, "no stop reason"),
        ({"stopReason": ""}, "no stop reason"),
        ({"usage": None}, "no whole token counts"),
        ({"usage": {"inputTokens": 640}}, "no whole token counts"),
        ({"usage": {"inputTokens": 640, "outputTokens": 1.5}}, "no whole token counts"),
        ({"usage": {"inputTokens": -1, "outputTokens": 2}}, "no whole token counts"),
        ({"usage": {"inputTokens": True, "outputTokens": 2}}, "no whole token counts"),
    ],
)
def test_a_malformed_response_is_a_failure(changes, message):
    with pytest.raises(PredictorError, match=message):
        read_response(response(**changes), 0.1)
    with pytest.raises(PredictorError, match="not a response"):
        read_response(None, 0.1)


@pytest.mark.parametrize("name", ["cacheReadInputTokens", "cacheWriteInputTokens"])
def test_cached_tokens_are_a_failure_because_the_input_count_is_then_partial(name):
    usage = {"inputTokens": 100, "outputTokens": 17, "totalTokens": 657, name: 540}
    with pytest.raises(PredictorError, match=f"reports cached tokens \\({name}\\)"):
        read_response(response(usage=usage), 0.1)
    usage[name] = 0  # reported as zero: nothing was cached
    assert read_response(response(usage=usage), 0.1).input_tokens == 100


# --------------------------------------------------------------------------- failures


@pytest.mark.parametrize("code", ERROR_CODES)
def test_each_bedrock_error_becomes_a_predictor_error_with_the_code_and_no_text(cfg, code):
    # A service message could repeat part of a request; it must not reach the error.
    error = client_error(code, message=f"bad input near: {USER}")
    provider, client = provider_with(cfg, error=error)
    with pytest.raises(PredictorError) as caught:
        provider.converse(MODEL, SYSTEM, USER, 192)
    message = str(caught.value)
    assert code in message and MODEL in message
    assert "kelp" not in message and "zebrafish" not in message and "bad input" not in message
    assert client.converse.call_count == 1  # retrying is botocore's, inside the client


@pytest.mark.parametrize("error", AUTH_ERRORS, ids=lambda error: type(error).__name__)
def test_a_missing_sign_in_names_the_command(cfg, error):
    provider, _ = provider_with(cfg, error=error)
    with pytest.raises(PredictorError, match="aws sso login --profile qlora"):
        provider.converse(MODEL, SYSTEM, USER, 192)


@pytest.mark.parametrize("code", EXPIRED_CODES)
def test_an_expired_sign_in_code_names_the_command(cfg, code):
    provider, _ = provider_with(cfg, error=client_error(code))
    with pytest.raises(PredictorError, match="aws sso login --profile qlora"):
        provider.converse(MODEL, SYSTEM, USER, 192)


@pytest.mark.parametrize(
    "error",
    [
        ReadTimeoutError(endpoint_url="https://stub.invalid"),
        EndpointConnectionError(endpoint_url="https://stub.invalid"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_a_timeout_or_no_network_is_a_predictor_error_naming_the_kind(cfg, error):
    provider, _ = provider_with(cfg, error=error)
    with pytest.raises(PredictorError, match=type(error).__name__) as caught:
        provider.converse(MODEL, SYSTEM, USER, 192)
    assert "stub.invalid" not in str(caught.value)


def test_a_malformed_response_from_the_client_is_a_predictor_error(cfg):
    provider, _ = provider_with(cfg, result={"output": {}})
    with pytest.raises(PredictorError, match="no assistant message"):
        provider.converse(MODEL, SYSTEM, USER, 192)


# --------------------------------------------------------------------------- the client


def test_client_config_comes_from_the_config_file(cfg):
    config = client_config(cfg)
    # total_max_attempts counts the first try; max_attempts would count retries only.
    assert config.retries == {"total_max_attempts": 12, "mode": "adaptive"}
    assert (config.connect_timeout, config.read_timeout) == (10, 120)


def test_the_runtime_client_is_built_from_the_session_with_that_config(cfg):
    session = MagicMock()
    client = bedrock_runtime_client(AWS, cfg, session=session)
    assert client is session.client.return_value
    (service,), keywords = session.client.call_args
    assert service == "bedrock-runtime"
    assert keywords["config"].retries == {"total_max_attempts": 12, "mode": "adaptive"}


def test_the_session_uses_the_configured_profile_and_region(monkeypatch):
    seen = {}

    def factory(profile_name=None, region_name=None):
        seen.update(profile=profile_name, region=region_name)
        return "session"

    monkeypatch.setattr(aws_common.boto3, "Session", factory)  # where the session is made
    assert aws_session(AWS) == "session"
    assert seen == {"profile": "qlora", "region": "us-east-1"}


def test_a_profile_that_does_not_exist_is_a_config_error_with_the_command():
    aws = dict(AWS, profile="no-such-profile-for-this-test")
    with pytest.raises(ConfigError, match="aws sso login --profile no-such-profile-for-this-test"):
        aws_session(aws)  # raised when the session is created: no call reaches AWS


# --------------------------------------------------------------------------- the sign-in check


def sts_with(account=None, error=None):
    sts = MagicMock()
    if error is not None:
        sts.get_caller_identity.side_effect = error
    else:
        sts.get_caller_identity.return_value = {"Account": account}
    return sts


def test_identity_passes_for_the_configured_account():
    sts = sts_with("123456789012")
    assert check_identity(sts, AWS) is None
    sts.get_caller_identity.assert_called_once_with()


def test_identity_refuses_another_account():
    with pytest.raises(ConfigError, match="in to account 111122223333, expected 123456789012"):
        check_identity(sts_with("111122223333"), AWS)


@pytest.mark.parametrize(
    "error",
    [*AUTH_ERRORS, *(client_error(code, operation="GetCallerIdentity") for code in EXPIRED_CODES)],
    ids=lambda error: type(error).__name__,
)
def test_identity_without_a_sign_in_is_a_config_error_with_the_command(error):
    with pytest.raises(ConfigError, match="aws sso login --profile qlora"):
        check_identity(sts_with(error=error), AWS)


def test_identity_reports_other_aws_errors_by_code_or_kind():
    with pytest.raises(ConfigError, match="did not confirm the sign-in: AccessDenied"):
        check_identity(sts_with(error=client_error("AccessDenied")), AWS)
    unreachable = EndpointConnectionError(endpoint_url="https://stub.invalid")
    with pytest.raises(ConfigError, match="could not be reached: EndpointConnectionError"):
        check_identity(sts_with(error=unreachable), AWS)
