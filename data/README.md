# Data and provenance

The benchmark inputs are not bundled in this public staging repository. This keeps the release free of private files and avoids implying that a paper-table value can be regenerated without the authors' frozen input package.

For a benchmark rerun, provide one record per dataset with:

```text
dataset_id,dataset,n_samples,n_features,n_classes,partition_file,labels_file,sha256,protocol
```

The partition file must contain the frozen base-partition labels used to build `X0`; the target labels are used only after filtering for external metrics. Each partition must have one label per sample and may use arbitrary label names. `begf.build_x0` converts the labels to deterministic nonempty-cluster indicator blocks.

`paper_results.csv` is a transparent transcription of Table 2 in the final paper PDF. It reports percentages, uses the final paper's ten dataset names and dimensions, and leaves unavailable valid outputs empty with `valid=NO`; no missing value is imputed. It is a result record, not a claim that the private frozen benchmark files are included here.
