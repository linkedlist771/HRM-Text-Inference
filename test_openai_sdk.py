"""Verify /v1/chat/completions works through the official OpenAI Python SDK,
both non-streaming and streaming, using `messages` (as real clients do)."""

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:1919/v1", api_key="not-needed")

print("== models ==")
print([m.id for m in client.models.list().data])

print("\n== non-streaming chat.completions ==")
r = client.chat.completions.create(
    model="hrm",
    messages=[{"role": "user", "content": "9.8 and 9.11, which is bigger?"}],
    max_tokens=40,
    temperature=0.0,
)
print("id/object/model:", r.id, "/", r.object, "/", r.model)
print("finish_reason:", r.choices[0].finish_reason, "| usage:", r.usage.completion_tokens, "tok")
print("content:", repr(r.choices[0].message.content[:100]))

print("\n== streaming chat.completions ==")
stream = client.chat.completions.create(
    model="hrm",
    messages=[{"role": "user", "content": "Explain why the sky is blue."}],
    max_tokens=40,
    temperature=0.0,
    stream=True,
)
buf, n = "", 0
for chunk in stream:
    assert chunk.object == "chat.completion.chunk", chunk.object
    delta = chunk.choices[0].delta
    if delta.content:
        buf += delta.content
        n += 1
print("chunks:", n, "| content:", repr(buf[:100]))

print("\nOPENAI SDK COMPATIBILITY: OK")
