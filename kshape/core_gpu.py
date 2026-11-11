import math
import torch
import numpy as np
from sklearn.base import ClusterMixin, BaseEstimator

# Upper bound on the number of elements of the temporary cross-correlation tensors; larger inputs are processed
# in chunks of series.
_CHUNK_ELEMENTS = 1 << 27
# eigh is called once for all (cluster, dimension) problems when they are small or many; otherwise once per
# problem, which is faster for a few large matrices.
_SMALL_EIGH = 256
_BATCHED_EIGH_MIN = 6


def zscore(a, axis=0, ddof=0):
    mns = a.mean(dim=axis)
    sstd = a.std(dim=axis, unbiased=(ddof == 1))
    if axis and mns.dim() < a.dim():
        x=(a - mns.unsqueeze(axis)).div(sstd.unsqueeze(axis))
        return x.masked_fill(torch.isnan(x), 0)
        #return torch.nan_to_num((a - mns.unsqueeze(axis)).div(sstd.unsqueeze(axis)))
    else:
        x=a.sub_(mns).div(sstd)
        return x.masked_fill(torch.isnan(x), 0)
        #return torch.nan_to_num(a.sub_(mns).div(sstd))


def roll_zeropad(a, shift, axis=None):
    if shift == 0:
        return a
    if abs(shift) > len(a):
        return torch.zeros_like(a)

    padding = torch.zeros(abs(shift), a.shape[1], device="cuda", dtype=torch.float32)
    if shift < 0:
        return torch.cat((a[abs(shift):], padding))
    else:
        return torch.cat((padding, a[:-shift]))


def _ncc_c_3dim(x, y):
    den = torch.norm(x, p=2, dim=(0, 1)) * torch.norm(y, p=2, dim=(0, 1))

    if den < 1e-9:
        den = torch.tensor(float("inf"), device="cuda", dtype=torch.float32)

    x_len = x.shape[0]
    fft_size = 1 << (2*x_len-1).bit_length()

    cc = torch.fft.ifft(torch.fft.fft(x, fft_size, dim=0) * torch.conj(torch.fft.fft(y, fft_size, dim=0)), dim=0)
    cc = torch.cat((cc[-(x_len-1):], cc[:x_len]), dim=0)

    return torch.div(torch.sum(torch.real(cc), dim=-1), den)


def _sbd(x, y):
    ncc = _ncc_c_3dim(x, y)
    idx = ncc.argmax().item()
    yshift = roll_zeropad(y, (idx + 1) - max(len(x), len(y)))

    return yshift


def _zscore_dim(a, dim):
    """z-normalises along dim with ddof=1; constant slices become zero."""
    mu = a.mean(dim=dim, keepdim=True)
    sd = a.std(dim=dim, keepdim=True, unbiased=True)
    return torch.nan_to_num((a - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)


def _lags(x_len, fft_size, device):
    """Positions of the lags -(x_len-1) .. x_len-1 in a circular cross-correlation of length fft_size."""
    return torch.cat((torch.arange(fft_size - (x_len - 1), fft_size), torch.arange(0, x_len))).to(device)


class _Batched:
    """Per-dataset state of the batched k-Shape: the spectrum and the norms of every series, computed once."""

    def __init__(self, x):
        self.x = x  # (m, L, D)
        m, L, D = x.shape
        self.m, self.L, self.D = m, L, D
        self.fft_size = 1 << (2 * L - 1).bit_length()
        self.lags = _lags(L, self.fft_size, x.device)
        self.fx = torch.fft.rfft(x, n=self.fft_size, dim=1)  # (m, fft_size/2+1, D)
        self.xnorm = torch.linalg.vector_norm(x, dim=1)  # (m, D)
        self.xfro = torch.linalg.vector_norm(x, dim=(1, 2))  # (m,)

    def align(self, idx, centroids):
        """SBD-aligns every series to the centroid of its cluster, independently per dimension (as _sbd does)."""
        L, D, n = self.L, self.D, self.fft_size
        fc = torch.fft.rfft(centroids, n=n, dim=1)
        cnorm = torch.linalg.vector_norm(centroids, dim=1)  # (k, D)
        zero = (centroids == 0).all(dim=1)  # (k, D): series are not shifted towards an all-zero centroid
        shift = torch.empty(self.m, D, dtype=torch.long, device=self.x.device)
        step = max(1, _CHUNK_ELEMENTS // (n * D))
        for s in range(0, self.m, step):
            e = min(self.m, s + step)
            lab = idx[s:e]
            cc = torch.fft.irfft(fc[lab] * self.fx[s:e].conj(), n=n, dim=1).index_select(1, self.lags)
            arg = cc.argmax(dim=1)
            arg = torch.where(cnorm[lab] * self.xnorm[s:e] >= 1e-9, arg, torch.zeros_like(arg))
            shift[s:e] = torch.where(zero[lab], torch.zeros_like(arg), arg + 1 - L)
        t = torch.arange(L, device=self.x.device).view(1, L, 1)
        src = t - shift.view(self.m, 1, D)
        valid = (src >= 0) & (src < L)
        return torch.gather(self.x, 1, src.clamp(0, L - 1)) * valid

    def distances(self, centroids):
        """1 - max NCC between every series and every centroid, summed over dimensions (as _ncc_c_3dim does)."""
        n = self.fft_size
        k = centroids.shape[0]
        fc = torch.fft.rfft(centroids, n=n, dim=1).conj()
        cfro = torch.linalg.vector_norm(centroids, dim=(1, 2))
        best = torch.empty(self.m, k, dtype=self.x.dtype, device=self.x.device)
        step = max(1, _CHUNK_ELEMENTS // (k * n))
        for s in range(0, self.m, step):
            e = min(self.m, s + step)
            spec = torch.einsum("ifd,jfd->ijf", self.fx[s:e], fc)
            best[s:e] = torch.fft.irfft(spec, n=n, dim=2).index_select(2, self.lags).max(dim=2).values
        den = self.xfro.view(-1, 1) * cfro.view(1, -1)
        return 1 - torch.where(den < 1e-9, torch.zeros_like(best), best / den)


def _shape_vectors(y, idx, k):
    """Top eigenvector of P S P for every (cluster, dimension), S = Y^T Y over the cluster's z-normalised members.

    The rows of Y have zero mean, so P S P = S, and the top eigenvector of Y^T Y is Y^T u for the top eigenvector u
    of the Gram matrix Y Y^T. Whichever of the two is smaller is solved. Usually all (cluster, dimension) problems
    go to one batched eigh call (Gram matrices zero-padded to the largest cluster, which only adds zero eigenvalues);
    a few large problems are solved one at a time, which is faster than a small batch of large matrices.
    Returns (k, L, D) unit vectors (zero for empty clusters) and the cluster sizes.
    """
    m, L, D = y.shape
    counts = torch.bincount(idx, minlength=k).tolist()
    order = torch.argsort(idx, stable=True)
    groups = []
    start = 0
    for j in range(k):
        groups.append(y.index_select(0, order[start:start + counts[j]]))
        start += counts[j]
    vec = torch.zeros(k, L, D, dtype=y.dtype, device=y.device)
    nonempty = [j for j in range(k) if counts[j] > 0]
    if not nonempty:
        return vec, counts

    def from_gram(yj):  # yj: (..., n, L, D)
        _, u = torch.linalg.eigh(torch.einsum("...nld,...qld->...dnq", yj, yj))
        v = torch.einsum("...nld,...dn->...ld", yj, u[..., -1])
        return v / torch.linalg.vector_norm(v, dim=-2, keepdim=True).clamp_min(1e-30)

    def cov(yj):  # (n, L, D) -> (D, L, L)
        return torch.einsum("nld,nsd->dls", yj, yj)

    def from_cov(s):  # (..., D, L, L) -> (..., L, D)
        _, w = torch.linalg.eigh(s)
        return w[..., -1].transpose(-1, -2)

    r = max(min(counts[j], L) for j in nonempty)
    if r <= _SMALL_EIGH or len(nonempty) * D >= _BATCHED_EIGH_MIN:
        n = max(counts[j] for j in nonempty)
        if n < L:
            yp = torch.zeros(len(nonempty), n, L, D, dtype=y.dtype, device=y.device)
            for g, j in enumerate(nonempty):
                yp[g, :counts[j]] = groups[j]
            vec[nonempty] = from_gram(yp)
        else:
            vec[nonempty] = from_cov(torch.stack([cov(groups[j]) for j in nonempty]))
    else:
        for j in nonempty:
            for d in range(D):
                yjd = groups[j][..., d:d + 1]
                vec[j, :, d:d + 1] = from_gram(yjd) if counts[j] < L else from_cov(cov(yjd))
    return vec, counts


def _extract_shape(idx, x, j, cur_center):
    _a = []
    for i in range(len(idx)):
        if idx[i] == j:
            if torch.all(cur_center == 0):
                opt_x = x[i]
            else:
                opt_x = _sbd(cur_center, x[i])
            _a.append(opt_x)

    if len(_a) == 0:
        indices = torch.randperm(x.shape[0])[:1]
        return torch.squeeze(x[indices].clone())
        #return torch.zeros((x.shape[1]))

    a = torch.stack(_a)

    columns = a.shape[1]
    y = zscore(a, axis=1, ddof=1)

    s = y[:, :, 0].transpose(0, 1).mm(y[:, :, 0])

    p = torch.empty((columns, columns), device="cuda", dtype=torch.float32)
    p.fill_(1.0 / columns)
    p = torch.eye(columns, device="cuda", dtype=torch.float32) - p

    m = p.mm(s).mm(p)
    _, vec = torch.linalg.eigh(m,UPLO='U')
    centroid = vec[:, -1]

    finddistance1 = torch.norm(a.sub(centroid.reshape((x.shape[1], 1))), 2, dim=(1, 2)).sum()
    finddistance2 = torch.norm(a.add(centroid.reshape((x.shape[1], 1))), 2, dim=(1, 2)).sum()

    if finddistance1 >= finddistance2:
        centroid.mul_(-1)

    return zscore(centroid, ddof=1)


def _kshape(x, k, centroid_init='zero', max_iter=100, batched=True):
    m = x.shape[0]
    idx = torch.randint(0, k, (m,), dtype=torch.float32).to("cuda")
    if centroid_init == 'zero':
        centroids = torch.zeros(k, x.shape[1], x.shape[2], device="cuda", dtype=torch.float32)
    elif centroid_init == 'random':
        indices = torch.randperm(x.shape[0])[:k]
        centroids = x[indices].clone()
    if not batched:
        return _kshape_loop(x, k, idx, centroids, max_iter)
    idx = idx.long()
    data = _Batched(x)

    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1
        old_idx = idx
        # shape extraction for all clusters and dimensions at once
        a = data.align(idx, centroids)
        y = _zscore_dim(a, dim=1)
        vec, counts = _shape_vectors(y, idx, k)
        # keep the orientation closer to the aligned members
        v = vec.index_select(0, idx)
        d1 = torch.zeros(k, x.shape[2], device=x.device, dtype=x.dtype).index_add_(0, idx, torch.linalg.vector_norm(a - v, dim=1))
        d2 = torch.zeros(k, x.shape[2], device=x.device, dtype=x.dtype).index_add_(0, idx, torch.linalg.vector_norm(a + v, dim=1))
        centroids = _zscore_dim(torch.where((d1 >= d2).unsqueeze(1), -vec, vec), dim=1)
        for j in range(k):
            if counts[j] == 0:
                for d in range(x.shape[2]):
                    indices = torch.randperm(x.shape[0])[:1]
                    centroids[j, :, d] = torch.squeeze(x[indices, :, d].clone())

        idx = data.distances(centroids).argmin(1)
        if torch.equal(old_idx, idx):
            break

    return idx, centroids, n_iter


def _kshape_loop(x, k, idx, centroids, max_iter):
    """The per-series version: one shape extraction per (cluster, dimension), one NCC per (series, centroid)."""
    m = x.shape[0]
    distances = torch.empty(m, k, device="cuda")

    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1
        old_idx = idx
        for j in range(k):
            for d in range(x.shape[2]):
                centroids[j, :, d] = _extract_shape(idx, torch.unsqueeze(x[:, :, d], axis=2), j, torch.unsqueeze(centroids[j, :, d], axis=1))
                #centroids[j] = torch.unsqueeze(_extract_shape(idx, x, j, centroids[j]), dim=1)

        for i, ts in enumerate(x):
            for c, ct in enumerate(centroids):
                dist = 1 - _ncc_c_3dim(ts, ct).max()
                distances[i, c] = dist

        idx = distances.argmin(1)
        if torch.equal(old_idx, idx):
            break

    return idx, centroids, n_iter


def _clusters(idx, centroids):
    return [(centroid, torch.nonzero(idx == i).flatten().tolist()) for i, centroid in enumerate(centroids)]


def kshape(x, k, centroid_init='zero', max_iter=100, batched=True):
    x = torch.tensor(x, device="cuda", dtype=torch.float32)
    idx, centroids, _ = _kshape(x, k, centroid_init=centroid_init, max_iter=max_iter, batched=batched)
    return _clusters(idx, centroids)

class KShapeClusteringGPU(ClusterMixin,BaseEstimator):
    """k-Shape on a CUDA device.

    With batched=True (the default), every step runs on all series and clusters at once. With batched=False, the
    shapes and distances are computed one series and one (series, centroid) pair at a time.
    """
    labels_= None
    centroids_ = None

    def __init__(self,n_clusters, centroid_init='zero', max_iter=100, batched=True):
        self.n_clusters = n_clusters
        self.centroid_init = centroid_init
        self.max_iter = max_iter
        self.batched = batched


    def fit(self,X,y=None):
        clusters = self._fit(X,self.n_clusters, self.centroid_init, self.max_iter)
        self.labels_ = np.zeros(X.shape[0])
        self.centroids_ =torch.zeros(self.n_clusters, X.shape[1], X.shape[2], device="cuda", dtype=torch.float32)
        for i in range(self.n_clusters):
            self.labels_[clusters[i][1]] = i
            self.centroids_[i]=clusters[i][0]
        return self

    def predict(self, X):
        labels, _ = self._predict(X,self.centroids_)
        return labels


    def _predict(self,x, centroids):
        x = torch.tensor(x, device="cuda", dtype=torch.float32)
        if self.batched:
            return _Batched(x).distances(centroids).argmin(1), centroids
        m = x.shape[0]
        distances = torch.empty(m, self.n_clusters, device="cuda")

        for i, ts in enumerate(x):
            for c, ct in enumerate(centroids):
                dist = 1 - _ncc_c_3dim(ts, ct).max()
                distances[i, c] = dist

        idx = distances.argmin(1)
        return idx, centroids


    def _fit(self,x, k, centroid_init='zero', max_iter=100):
        x = torch.tensor(x, device="cuda", dtype=torch.float32)
        idx, centroids, self.n_iter_ = _kshape(x, k, centroid_init=centroid_init, max_iter=max_iter, batched=self.batched)
        return _clusters(idx, centroids)



if __name__ == "__main__":
    import sys
    import doctest
    sys.exit(doctest.testmod()[0])
