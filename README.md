# source-Imaging

A collection of scripts and tools for EEG source imaging using MNE-Python. This repository encompasses preprocessing pipelines, source localization methods, statistical analysis, and visualization utilities tailored for EEG data analysis.([github.com][1])

## Features

* **Preprocessing Pipelines**: Automated scripts for EEG data cleaning, including bad channel detection and artifact rejection.
* **Source Localization**: Implementations of various source imaging techniques using MNE-Python.
* **Statistical Analysis**: Tools for cluster-based permutation tests and other statistical evaluations.
* **Visualization**: Scripts for plotting sensor layouts, source estimates, and statistical results.
* **Compatibility**: Supports data formats commonly used in EEG research, such as `.mat` and `.vmrk` files.([researchgate.net][2], [github.com][1])

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/berdakh/source-Imaging.git
   cd source-Imaging
   ```



2. **Create a virtual environment (optional but recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```



3. **Install required packages**:

   ```bash
   pip install -r requirements.txt
   ```



*Note: Ensure that [MNE-Python](https://mne.tools/stable/index.html) is installed, as it's central to the functionalities provided.*

## Usage

The repository contains multiple scripts, each serving specific purposes. Here's a brief overview:

* **Preprocessing**:

  * `preproc.py`, `batch_preproc.py`: Scripts for preprocessing EEG data, including filtering and artifact removal.
  * `badchannel-example.py`: Example script for detecting and handling bad channels.([github.com][1])

* **Source Localization**:

  * `plot_mne_dspm_source_localization.py`: Demonstrates source localization using dSPM method.
  * `uh_forward.py`, `uh_inverseBatch.py`: Scripts for forward and inverse modeling.([github.com][1])

* **Statistical Analysis**:

  * `plot_stats_cluster_spatio_temporal.py`, `stats_cluster_spatio_temporal.py`: Perform cluster-based permutation tests.([github.com][1])

* **Visualization**:

  * `plot_forward.py`, `plot_source_alignment.py`: Visualize forward models and source alignments.([github.com][1])

* **Utilities**:

  * `find_vmrk.py`: Locate and process `.vmrk` files.
  * `load_mat_MRCP.py`: Load `.mat` files containing MRCP data.([github.com][1])

*To execute a script, navigate to the repository directory and run:*

```bash
python script_name.py
```



*Replace `script_name.py` with the desired script.*

## File Structure

```plaintext
source-Imaging/
├── preprocessing/
│   ├── preproc.py
│   ├── batch_preproc.py
│   └── badchannel-example.py
├── source_localization/
│   ├── plot_mne_dspm_source_localization.py
│   ├── uh_forward.py
│   └── uh_inverseBatch.py
├── statistics/
│   ├── plot_stats_cluster_spatio_temporal.py
│   └── stats_cluster_spatio_temporal.py
├── visualization/
│   ├── plot_forward.py
│   └── plot_source_alignment.py
├── utilities/
│   ├── find_vmrk.py
│   └── load_mat_MRCP.py
├── requirements.txt
└── README.md
```



*Note: The above structure is a suggested organization. Adjust directories as needed based on actual file locations.*

## Open-weight models (`eegmodels`)

The `eegmodels/` package is a unified downloader and loader for open-weight EEG
foundation models (braindecode zoo, LaBraM, REVE, ...). It keeps a registry of models
with their source, license and input contract, downloads weights into a local cache,
and runs them directly on MNE `Raw`/`Epochs` objects:

```bash
pip install -e ".[braindecode,mne]"
eegmodels list
eegmodels pull steegformer-base
```

See [`eegmodels/README.md`](eegmodels/README.md) for the design and how to add a model.

## Contributing

Contributions are welcome! If you have suggestions, bug reports, or enhancements, please open an issue or submit a pull request.([github.com][3])

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments

This repository was developed by [Berdakh Abibullaev](https://github.com/berdakh), focusing on EEG source imaging techniques using MNE-Python.([github.com][4])

---

*For detailed explanations and methodologies, refer to the [Source localization with MNE.pdf](Source%20localization%20with%20MNE.pdf) included in the repository.*
