"""v2 client quickstart — one SDK for native inference and management.

The MLPal client speaks the native Anthropic Messages wire for inference and the
/admin/v1 control plane for administration. Point base_url at the managed
service (default) or a self-hosted gateway — the surface is identical.

    export MLPAL_API_KEY=mlpal_sk_...
    # optional: export MLPAL_BASE_URL=http://localhost:8000   # self-hosted
    python examples/v2_quickstart.py
"""

from __future__ import annotations

from mlpal_assistants import MLPal


def main() -> None:
    with MLPal() as client:
        # ── native inference ────────────────────────────────────────────────
        # `mlpal` is a meta-model: the gateway routes it to the best-quality
        # model. Any concrete tag (claude-opus-5, gpt-5.6, gemini-3-pro, or a
        # custom adapter) works here too — the response always comes back in
        # this Anthropic shape.
        msg = client.messages.create(
            model="mlpal",
            max_tokens=512,
            system="You are concise.",
            messages=[{"role": "user", "content": "Give me one fact about otters."}],
        )
        print("assistant:", msg.text)
        print("usage:", msg.usage)

        # ── streaming ───────────────────────────────────────────────────────
        print("\nstreaming: ", end="", flush=True)
        with client.messages.stream(
            model="mlpal",
            max_tokens=512,
            messages=[{"role": "user", "content": "Count to five."}],
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    delta = event.data.get("delta", {})
                    print(delta.get("text", ""), end="", flush=True)
        print()

        # ── management (admin key required) ─────────────────────────────────
        # Issue a scoped inference key: model policy + a monthly spend cap.
        # Requires an admin-scoped principal (Cognito JWT on the managed
        # service, or an admin API key when self-hosting).
        try:
            key = client.admin.keys.create(
                name="team-web",
                permissions=["messages"],
                model_policy={"allow": ["claude-*", "mlpal*"], "deny": []},
                budgets=[{"unit": "usd", "amount": 100, "window": "month"}],
            )
            print(f"\ncreated key {key.id} (secret shown once): {key.secret[:16]}…")
        except Exception as e:  # noqa: BLE001 — this key may not be admin-scoped
            print(f"\n(management skipped: {e})")


if __name__ == "__main__":
    main()
