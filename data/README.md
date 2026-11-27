# Data supplied for the study

`raw/sensors/Laboratory.csv` preserves the original semicolon-delimited qualified laboratory results; `Laboratory_numeric.csv` supplies verified finer numeric precision. `SCADA.csv` holds hourly intake measurements and `Weather_raw.csv` the archived local-station records. Sources are the water utility and its contracted laboratory; units and original qualifiers are retained.

`FullHourly_source_corrected.csv` and `Profiles_source_corrected.csv` are the reconciled logger records before sensor/depth QC. Documented import zeros corresponding to missing logger tokens have been restored to missing; genuine zeros and finite precision are retained. `missingness_corrections.csv`, `missingness_summary.csv` and `profiler_source.json` record that source-level reconciliation. The workflow performs sensor screening and depth selection from these included records; it does not use the former zero-filled imports.

`raw/met_nordic/` contains the MET Nordic long-term consistent point archive covering the station record and its download manifest. MET Norway data are supplied under CC BY 4.0; credit the Norwegian Meteorological Institute. The manifest records archive requests and point extraction. Figure 5c uses raw sources; model gap filling uses only cutoff-specific corrected values.

`derived/` is rebuilt locally and is not a scientific input archive. Its manifest hashes sources and prepared outputs. Native logger timestamps retain their recorded wall clock. Laboratory noon timestamps are a modelling convention, not verified sample-collection or reporting times.

The software licence does not grant additional rights over third-party records; retain the source attributions and applicable data terms.

The authors confirmed on 21 November 2026 that the utility and laboratory records are cleared for public release. They are supplied with permission. The utility and laboratory records may be used to reproduce the methods and results of this study. Other reuse requires permission from the data owners; contact the corresponding author, Russell Primeau (russell.b.primeau@ntnu.no), to request it. This study-specific permission was confirmed by the authors on 27 November 2026. The software GPL licence does not extend to these records. MET Norway attribution and its CC BY 4.0 terms apply to that source.

The reproduction inputs intentionally exclude the older consolidated `raw/Laboratory.csv`,
`raw/Limits.csv`, `raw/sensors/Eurofins.csv`, `raw/sensors/FullHourly.csv` and
`raw/sensors/Weather_alt_source.csv` exports. Fresh preparation and reporting use
the qualified/numeric laboratory pair, reconciled profiler sources, intake record
and weather sources described above. Historical filenames in `profiler_source.json`
identify source-reconciliation provenance; those original development imports are
not additional runtime dependencies.
