# GNW Self-Hosted Monitoring

The GNW Self-Hosted Monitoring System allows for running post-2020 deforestation risk
assessment and disturbance-alert analyses on your local system, using datasets
provided by GNW.

For motivation and high-level description of the GNW Self-Hosted Monitoring System,
visit [GNW Self-Hosted
Monitoring Assets](https://gfw.atlassian.net/wiki/external/YzlkMTQ0OWY4MzM3NGYxZTljM2M3MmU1NDI5NmQwMDM)

For full technical documentation on running the analysis script `post2020.py` on a set of
geometries, see [Technical Documentation](https://gfw.atlassian.net/wiki/external/MzYxZDJlMzBjOTA4NDU0Njk2N2NiNTgzNDg1NDY1NjE).
The default analysis option directly accesses the data from the GNW S3 buckets. The
documention also includes a full description of the other supported configuration, where you
download the required datasets to your local environment for better reliability, greater
scalability, and lower access costs.  The documentation also describes the helper scripts
`check_updates.py` and `hash_zarr.py`.

These scripts are published under an Apache 2.0 license.
