# Prepare data with a prompt

The prompt is a convenience for writing two JSON files. The data remains one Parquet table plus its embedding arrays. It does not automatically discover semantics from column names.

| Walkthrough | Completed template | Worked response | Executed report |
| --- | --- | --- | --- |
| Customer measurements, category and boolean | [Prompt](numeric_categorical/prep_prompt.md) | [Both JSON files and decisions](numeric_categorical/prep_response.md) | [Report](numeric_categorical/audit_output/report.html) |
| Measurements and two disjoint cluster assignments | [Prompt](disjoint_clusters/prep_prompt.md) | [Both JSON files and decisions](disjoint_clusters/prep_response.md) | [Report](disjoint_clusters/audit_output/report.html) |

These are authored, reproducible reference responses, not a transcript of an external LLM execution. Their schema/config contents are the files actually used to run the examples.

## Apply the same workflow to your table

1. Run `parquet-audit inspect --parquet your_data.parquet --out inspection`.
2. Give `inspection/prepare_prompt.md` to your LLM together with the facts it cannot infer: item identity, column meanings, null/sentinel semantics, cluster versus tag semantics, and the embedding paths/dimensions/ID ordering. The two prompts above show what those additions look like.
3. Save its `dataset_schema.json` and `audit_config.json`, review the semantic choices, then run `parquet-audit run`.

The run performs the required type/alignment checks. There is no separate preparation-notes file or validation command. A successful run cannot establish that an LLM guessed the right meaning: that comes from the supplied domain facts.

To regenerate these walkthroughs from the actual prompt template and schemas, run `python examples/generate_prompt_examples.py` after generating the example data.
