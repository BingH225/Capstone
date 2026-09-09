# Source audit status

The JSON files in this directory contain only aggregate counts and SHA-256 fingerprints of the local source snapshots; no dialogue text is copied here.

Both CounselChat snapshots currently have `license_resolved: false`. They are therefore **not approved training sources**. Do not change the license string based on a dataset mirror. Before production use, identify the authoritative source terms for the exact file snapshot, record the evidence and review decision, then rerun the audit and data release pipeline.

The counts also show that deduplication and privacy review are mandatory: the snapshots contain repeated questions and pattern-detected URLs, phone-like strings, or handles. These are triage statistics, not proof that all personally identifiable information has been found.
