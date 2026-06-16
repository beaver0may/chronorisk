from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetCard:
    name: str
    version: str
    license: str
    url: str
    role: str
    access: str


_REGISTRY: tuple[DatasetCard, ...] = (
    DatasetCard(
        name="nscisc",
        version="public-use de-identified",
        license="NSCISC public-use de-identified terms",
        url="https://sites.uab.edu/nscisc/database/",
        role="primary SCI chronic-complication cohort",
        access="public",
    ),
    DatasetCard(
        name="mimic-iv",
        version="v3.1",
        license="PhysioNet Credentialed Health Data License 1.5.0",
        url="https://physionet.org/content/mimiciv/3.1/",
        role="temporal modeling validation",
        access="credentialed",
    ),
    DatasetCard(
        name="eicu-crd",
        version="v2.0",
        license="PhysioNet Credentialed Health Data License 1.5.0",
        url="https://physionet.org/content/eicu-crd/2.0/",
        role="cross-site generalization",
        access="credentialed",
    ),
    DatasetCard(
        name="uk-biobank",
        version="accelerometer sub-study",
        license="UK Biobank Access Management System approval",
        url="https://www.ukbiobank.ac.uk/",
        role="biosensor-EHR fusion validation",
        access="application",
    ),
    DatasetCard(
        name="all-of-us",
        version="v8 data release",
        license="All of Us Researcher Workbench controlled tier",
        url="https://allofus.nih.gov/",
        role="multimodal fusion at scale",
        access="controlled",
    ),
    DatasetCard(
        name="mimic-iv-ecg",
        version="v1.0",
        license="Open Data Commons Open Database License v1.0",
        url="https://physionet.org/content/mimic-iv-ecg/1.0/",
        role="ECG-EHR fusion",
        access="open",
    ),
    DatasetCard(
        name="epicare",
        version="NeurIPS 2024 Datasets and Benchmarks",
        license="open-source benchmark",
        url="https://github.com/Grosenick-Lab-Cornell",
        role="RL policy benchmark (simulated)",
        access="open",
    ),
)


def all_cards() -> tuple[DatasetCard, ...]:
    return _REGISTRY


def card(name: str) -> DatasetCard:
    for entry in _REGISTRY:
        if entry.name == name:
            return entry
    raise KeyError(f"unknown dataset '{name}'")
