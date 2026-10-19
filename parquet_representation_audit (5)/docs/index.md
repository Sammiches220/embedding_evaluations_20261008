# Documentation

Start with the [README](../README.md) for the three-step workflow. Open the deeper guide only for the question you have:

- [Data model and feature meanings](data-model.md): row alignment, feature kinds, missingness, type conversion, and information loss.
- [Information preservation and limits](information-preservation.md): direct mappings, lossy transformations, caps, sampling, and what probes cannot certify.
- [Probe models and controls](probes.md): XGBoost installation, shared model families, normalization configuration and metric colors.
- [Preparation prompt walkthroughs](../examples/prompt_walkthroughs.md): two completed prompts, worked JSON responses and executed reports.
- [Analysis and report guide](analysis.md): neighborhood scales, report measures, sampling scope, and topology.

The report is the main user interface. The implementation keeps ingestion, semantics, analysis, and report rendering in separate modules; most users only need the CLI and these guides.
