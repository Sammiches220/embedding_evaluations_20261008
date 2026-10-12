# Prepared synthetic bundle

All row/column semantics are explicitly declared in dataset_schema.json. This is generated demonstration data. Missing embeddings are intentionally present; invalid_embeddings=exclude records them. Scalar feature missingness does not remove other evaluation rows. No real-world performance claims apply.

IDs, category meanings, count units, tag delimiters and known-empty policy are documented by the JSON. Retail uses large integer IDs, embedded list vectors, explicit group-safe splits and Arrow dictionary categories. Telemetry uses leading-zero strings, pipe-delimited tags and shuffled external A/B IDs. Source-only disables all embedding analyses.
