import hashlib
import math

import numpy as np


BASE_SEED = 42
N_RANDOM_SETS = 1000


def protein_seed(uniprot_id):
    """
    Deterministic protein-specific uint32 seed.

    Scientific random stream depends ONLY on:
        base seed 42
        UniProt ID

    It does not depend on label, modality,
    attribution, prediction, conflict, or outcome.
    """

    material = (
        f"{BASE_SEED}|{uniprot_id}|"
        "DELETION_RANDOM_V1"
    ).encode("utf-8")

    digest = hashlib.sha256(
        material
    ).digest()

    return (
        int.from_bytes(
            digest[:8],
            byteorder="little",
            signed=False,
        )
        % (2**32)
    )


def deletion_k(length):
    """
    Frozen 10% joint-deletion budget.
    """

    length = int(length)

    if length <= 0:
        raise ValueError(
            "Protein length must be positive."
        )

    return max(
        1,
        math.ceil(
            0.10 * length
        ),
    )


def generate_random_sets(
    uniprot_id,
    length,
):
    """
    Generate exactly 1000 matched random
    deletion sets for one protein.

    Returns:
        int32 array of shape [1000, k]

    Each row:
        - contains exactly k positions
        - contains unique positions
        - uses zero-based residue indices
        - is sorted only for canonical storage/QC

    The same returned array must be used by
    sequence and structure analyses.
    """

    length = int(length)
    k = deletion_k(length)

    rng = np.random.default_rng(
        protein_seed(
            str(uniprot_id)
        )
    )

    sets = np.empty(
        (
            N_RANDOM_SETS,
            k,
        ),
        dtype=np.int32,
    )

    for i in range(
        N_RANDOM_SETS
    ):

        positions = rng.choice(
            length,
            size=k,
            replace=False,
        )

        positions.sort()

        sets[i] = positions.astype(
            np.int32,
            copy=False,
        )

    return sets


def validate_random_sets(
    sets,
    length,
):
    """
    Strict QC for the frozen random sets.
    """

    length = int(length)
    k = deletion_k(length)

    sets = np.asarray(
        sets
    )

    if sets.shape != (
        N_RANDOM_SETS,
        k,
    ):
        raise RuntimeError(
            "Random-set shape mismatch: "
            f"{sets.shape}; expected "
            f"({N_RANDOM_SETS}, {k})."
        )

    if not np.issubdtype(
        sets.dtype,
        np.integer,
    ):
        raise RuntimeError(
            "Random-set positions must "
            "be integer."
        )

    if sets.min() < 0:
        raise RuntimeError(
            "Random-set position < 0."
        )

    if sets.max() >= length:
        raise RuntimeError(
            "Random-set position out of bounds."
        )

    for row in sets:

        if len(
            np.unique(row)
        ) != k:
            raise RuntimeError(
                "Duplicate residue within "
                "random deletion set."
            )

    return True
