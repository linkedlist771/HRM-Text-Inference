"""Exact bf16 parity check: minisgl server vs HF reference (ref_bf16.json).

Run capture_ref_bf16() (server stopped) first to write ref_bf16.json, then
start the server and run this to compare token-for-token.
"""

import json
import urllib.request

REF = "ref_bf16.json"


def main():
    ref = json.load(open(REF))
    prompt = ref["prompt"]
    n = len(ref["gen_ids"])
    body = json.dumps(
        {"model": "hrm", "prompt": prompt, "max_tokens": n, "temperature": 0.0}
    ).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:1919/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    server_txt = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"][
        "content"
    ]
    hf_txt = ref["text"]
    print("=== HF bf16 reference ===")
    print(repr(hf_txt))
    print("=== minisgl server (bf16) ===")
    print(repr(server_txt))
    match = server_txt.strip() == hf_txt.strip()
    print(f"\nEXACT TEXT MATCH: {match}")
    raise SystemExit(0 if match else 1)


if __name__ == "__main__":
    main()
