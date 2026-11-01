"""PCA, whitening and shrinkage whitening, fitted per modality on Shopee TRAIN only.

One PCA for image vectors (512-d) and one for text vectors (768-d). A single PCA
on the concatenated vector would mix image and text in every component, so the
image-only and text-only cosines could no longer be read off it.
Data is centred, not standardised: every CLIP/MPNet dimension is on the same
scale already, and standardising would inflate low-variance noise dimensions.
"""
import numpy as np

VARIANTS = ("plain", "whiten", "shrink")
SHRINK = 0.10  # epsilon = 10% of the mean eigenvalue over all components


def fit_pca(X):
    """Mean, principal directions (rows, largest variance first) and eigenvalues of X.

    Computed by SVD of the centred data: if X - mean = U S V^T, the rows of V^T
    are the eigenvectors of the covariance and S^2 / (n - 1) its eigenvalues.
    """
    mean = X.mean(axis=0)
    _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
    return {"mean": mean, "components": vt, "eigvals": s ** 2 / (len(X) - 1)}


def normalize(X):
    """Scale each row to unit length, so a dot product is a cosine similarity."""
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def project(X, pca, dim, variant):
    """Reduce X to `dim` dimensions with one PCA variant and re-normalise.

    plain:  z = U_k^T (x - mu)
    whiten: z = Lambda_k^(-1/2) U_k^T (x - mu); every kept direction gets unit
            variance, so a few dominant directions no longer swamp the cosine.
    shrink: z = (Lambda_k + eps I)^(-1/2) U_k^T (x - mu); like whiten, but eps
            stops tiny-variance (often noisy) directions being blown up.
    Re-normalising is needed because centring and dropping dimensions change
    each vector's length, and FAISS inner product equals cosine only at unit length.
    dim=None means the raw vectors, no PCA.
    """
    if dim is None:
        return X
    z = (X - pca["mean"]) @ pca["components"][:dim].T
    lam = pca["eigvals"][:dim]
    if variant == "whiten":
        z = z / np.sqrt(lam)
    elif variant == "shrink":
        z = z / np.sqrt(lam + SHRINK * pca["eigvals"].mean())
    return normalize(z).astype(np.float32)


def explained_variance(pca, X=None):
    """Cumulative share of variance in the first 1..D components.

    Without X: on the train data the PCA was fitted on. With X: on new data
    projected with the frozen train mean and directions (the India shift check).
    """
    if X is None:
        per_component = pca["eigvals"]
    else:
        per_component = (((X - pca["mean"]) @ pca["components"].T) ** 2).mean(axis=0)
    return np.cumsum(per_component) / per_component.sum()
