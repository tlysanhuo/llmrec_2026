# e3_userres_r80_retkl_v3_s800 adapter

This is the current `s800` platform adapter that reached a best displayed
score of `1.0048` under the `platform-stable-v3.1-20260713` protocol.

The original `adapter_model.safetensors` is about 193 MB, so it is published
as three GitHub-compatible parts. The parts are byte-contiguous and must be
joined before loading the adapter.

## Restore

From this directory, run:

```bash
cat adapter_model.safetensors.part-aa \
    adapter_model.safetensors.part-ab \
    adapter_model.safetensors.part-ac \
    > adapter_model.safetensors
sha256sum -c SHA256SUMS
```

Load the restored `adapter_model.safetensors` together with
`adapter_config.json`. The expected full adapter SHA256 is:

```text
bb86eb8af0efd3560b7b7c8440f3830627e9255f4fcc2265b9274a27668f63c6
```

This package contains the adapter only; it does not include the base model.
