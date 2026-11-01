#!/usr/bin/env python
# -*- coding: utf-8 -*-

# *************************************************************************** #
#                  Copyright © 2022, UChicago Argonne, LLC                    #
#                           All Rights Reserved                               #
#                         Software Name: Tomocupy                             #
#                     By: Argonne National Laboratory                         #
#                                                                             #
#                           OPEN SOURCE LICENSE                               #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
# 1. Redistributions of source code must retain the above copyright notice,   #
#    this list of conditions and the following disclaimer.                    #
# 2. Redistributions in binary form must reproduce the above copyright        #
#    notice, this list of conditions and the following disclaimer in the      #
#    documentation and/or other materials provided with the distribution.     #
# 3. Neither the name of the copyright holder nor the names of its            #
#    contributors may be used to endorse or promote products derived          #
#    from this software without specific prior written permission.            #
#                                                                             #
#                                                                             #
# *************************************************************************** #
#                               DISCLAIMER                                    #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS         #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT           #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS           #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT    #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,      #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED    #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR      #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF      #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING        #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS          #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                #
# *************************************************************************** #

import numpy as np
import cupy as cp

__author__ = "Tomocupy contributors"
__copyright__ = "Copyright (c) 2022, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['FDKRec']


# One thread per output voxel, looping over the projections in the chunk and
# accumulating with += (so several chunks of a full angular range can be
# summed into the same volume chunk across multiple launches). Mirrors the
# indexing/accumulation style of kernels_linerec.cuh's backprojection_ker,
# generalized from parallel lines to divergent (cone-beam) rays. Geometry is
# the standard circular-trajectory, flat-panel FDK setup (Kak & Slaney,
# "Principles of Computerized Tomographic Imaging", ch. 3; Feldkamp, Davis &
# Kress 1984): source fixed at distance R from the rotation axis, detector at
# distance D from the source, both rotating rigidly with angle theta.
_KERNEL_SRC = r'''
extern "C" __global__
void fdk_backproj(
    float* __restrict__ f, const float* __restrict__ g, const float* __restrict__ theta,
    int n, int ncz, int nz_off, int nz_total,
    int nu, int nv, int ntp,
    float voxel_size, float du, float dv,
    float R, float D, float center_u, float center_v, float c)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
    if (ix >= n || iy >= n || iz >= ncz) return;

    float x = (ix - n*0.5f + 0.5f)*voxel_size;
    float y = (iy - n*0.5f + 0.5f)*voxel_size;
    float z = ((iz + nz_off) - nz_total*0.5f + 0.5f)*voxel_size;

    float acc = 0.0f;
    for (int t = 0; t < ntp; t++) {
        float beta = theta[t];
        float cb = cosf(beta), sb = sinf(beta);
        // rotate (x,y) into the source/detector frame of this projection:
        // s runs along the source-origin ray, tp perpendicular to it (-> u)
        float s  =  x*cb + y*sb;
        float tp = -x*sb + y*cb;
        float L = R - s;              // source-to-point distance along the ray axis
        if (L <= 1e-6f) continue;     // behind the source: no ray reaches it
        float invL = 1.0f/L;
        float u = D*tp*invL;
        float v = D*z*invL;
        float pu = u/du + nu*0.5f - 0.5f + center_u;
        float pv = v/dv + nv*0.5f - 0.5f + center_v;
        int iu = (int)floorf(pu);
        int ivv = (int)floorf(pv);
        if (iu < 0 || iu >= nu-1 || ivv < 0 || ivv >= nv-1) continue;
        float fu = pu-iu, fv = pv-ivv;
        long long base = (long long)t*nv*nu;
        float g00 = g[base + (long long)ivv*nu + iu];
        float g10 = g[base + (long long)ivv*nu + iu+1];
        float g01 = g[base + (long long)(ivv+1)*nu + iu];
        float g11 = g[base + (long long)(ivv+1)*nu + iu+1];
        float gi = g00*(1-fu)*(1-fv) + g10*fu*(1-fv) + g01*(1-fu)*fv + g11*fu*fv;
        float w = R*invL;             // standard FDK inverse-square weight (R/L)^2
        w *= w;
        acc += w*gi;
    }
    // (n-1-iy) flip to match the output orientation of the other tomocupy
    // reconstruction backends (fourierrec/linerec/lprec)
    long long fidx = (long long)iz*n*n + (long long)(n-1-iy)*n + ix;
    f[fidx] += acc*c;
}
'''


class FDKRec():
    """Cone-beam FDK reconstruction, circular trajectory, flat-panel detector.

    Geometry is handled in physical length units throughout (mm, or any
    consistent unit): source_origin_dist/source_detector_dist/du/dv/voxel_size
    must all share the same unit. There is no 180-degree-scan assumption and
    no coupling to the parallel-beam chunking in rec.py/rec_steps.py -- this
    is a self-contained geometry.
    """

    def __init__(self, nu, nv, n, nz, source_origin_dist, source_detector_dist,
                 du, dv, voxel_size, center_u=0.0, center_v=0.0, filter_name='parzen'):
        self.nu, self.nv, self.n, self.nz = int(nu), int(nv), int(n), int(nz)
        self.R = float(source_origin_dist)
        self.D = float(source_detector_dist)
        self.du, self.dv = float(du), float(dv)
        self.voxel_size = float(voxel_size)
        self.center_u, self.center_v = float(center_u), float(center_v)

        iu = cp.arange(self.nu, dtype='float32')
        ivv = cp.arange(self.nv, dtype='float32')
        uu = (iu - self.nu/2 + 0.5 - self.center_u)*self.du
        vv = (ivv - self.nv/2 + 0.5 - self.center_v)*self.dv
        # cosine (distance-to-source) pre-weighting, applied before filtering
        self.cos_weight = (self.R/cp.sqrt(self.R**2 + uu[cp.newaxis, :]**2 +
                                          vv[:, cp.newaxis]**2)).astype('float32')

        self.ne = 1
        while self.ne < 2*self.nu:
            self.ne *= 2
        self.pad = (self.ne - self.nu)//2
        self.wfilter = self._calc_filter(filter_name)
        self._kernel = cp.RawKernel(_KERNEL_SRC, 'fdk_backproj')

    def _wint(self, nw, t):
        """Band-limited ramp-filter quadrature design.

        Ported verbatim from reconstruction.fbp_filter.FBPFilter._wint: the
        math is purely 1D-filter design and does not depend on beam geometry,
        so it is reused as-is for the FDK ramp filter.
        """
        N = len(t)
        s = cp.linspace(1e-40, 1, nw)
        tmp1 = cp.arange(nw)
        tmp2 = cp.arange(1, nw+2)
        ivm = cp.linalg.inv(cp.exp(cp.outer(tmp1, cp.log(s))))
        u = cp.diff(cp.exp(cp.outer(tmp2, cp.log(s))) *
                    cp.tile(1.0/tmp2[..., cp.newaxis], [1, nw]))
        W1 = cp.matmul(ivm, u[1:nw+1, :])
        W2 = cp.matmul(ivm, u[0:nw, :])
        tmp1 = cp.arange(1, nw)
        tmp2b = (nw-1)*cp.ones((N-2*(nw-1)-1))
        tmp3 = cp.arange(nw-1, 0, -1)
        p = 1/cp.concatenate((tmp1, tmp2b, tmp3))
        w = cp.zeros(N)
        for j in range(N-nw+1):
            W = ((t[j+nw-1]-t[j])**2)*W1 + (t[j+nw-1]-t[j])*t[j]*W2
            for k in range(nw-1):
                w[j:j+nw] = w[j:j+nw] + p[j+k]*W[:, k]
        wn = w
        wn[-40:] = (w[-40])/(N-40)*cp.arange(N-40, N)
        return wn

    def _calc_filter(self, name):
        """Same filter shapes as FBPFilter.calc_filter, minus its final
        "/n". That division there folds in an inverse-FFT normalization
        needed by tomocupy's compiled (unnormalized) FFT filter kernel;
        filtering here instead uses cp.fft.irfft, which already normalizes,
        so repeating that division would apply it twice.
        """
        ne = self.ne
        d = 0.5
        t = cp.fft.rfftfreq(ne).astype('float64')
        if name == 'none':
            wfa = ne*0.5+t*0
            wfa[0] *= 2
        elif name == 'ramp':
            wfa = ne*0.5*self._wint(12, t)
        elif name == 'shepp':
            wfa = ne*0.5*self._wint(12, t)*cp.sinc(t/(2*d))*(t/d <= 2)
        elif name == 'cosine':
            wfa = ne*0.5*self._wint(12, t)*cp.cos(cp.pi*t/(2*d))*(t/d <= 1)
        elif name == 'cosine2':
            wfa = ne*0.5*self._wint(12, t)*(cp.cos(cp.pi*t/(2*d)))**2*(t/d <= 1)
        elif name == 'hamming':
            wfa = ne*0.5*self._wint(12, t)*(.54 + .46*cp.cos(cp.pi*t/d))*(t/d <= 1)
        elif name == 'hann':
            wfa = ne*0.5*self._wint(12, t)*(1+cp.cos(cp.pi*t/d))/2.0*(t/d <= 1)
        elif name == 'parzen':
            wfa = ne*0.5*self._wint(12, t)*pow(1-t/d, 3)*(t/d <= 1)
        else:
            raise ValueError(f'unknown fdk filter {name}')
        wfa = 2*wfa*(wfa >= 0)
        wfa[0] *= 2
        return wfa.astype('float32')

    def fan_angle(self, gamma_max_only=False):
        """Fan angle gamma = atan(u/D) for every detector column (radians).

        With gamma_max_only=True, returns only the largest |gamma| across the
        detector (the half-fan angle used by Parker weighting).
        """
        iu = cp.arange(self.nu, dtype='float32')
        uu = (iu - self.nu/2 + 0.5 - self.center_u)*self.du
        gamma = cp.arctan2(uu, cp.float32(self.D))
        if gamma_max_only:
            return float(cp.abs(gamma).max())
        return gamma

    def parker_weight(self, data, beta, scan_range):
        """Multiply by the Parker (1982) short-scan redundancy weight.

        This is the standard row-wise approximation used by practical
        cone-beam FDK implementations: the weight depends only on the
        projection angle and the detector column (fan angle gamma =
        atan(u/D)), and is applied identically to every detector row.

        data: [ntheta, nv, nu] float32 (GPU), in place
        beta: [ntheta] float32 (GPU), projection angle relative to the start
              of the scan (i.e. theta - theta[0]), radians, in [0, scan_range]
        scan_range: total covered angular range (radians); must be < 2*pi
        """
        gamma = self.fan_angle()                      # [nu]
        gamma_m = float(cp.abs(gamma).max())
        b = beta.astype('float32')[:, cp.newaxis]      # [ntheta, 1]
        g = gamma[cp.newaxis, :]                        # [1, nu]
        gm_minus_g = cp.maximum(gamma_m - g, 1e-6)
        gm_plus_g = cp.maximum(gamma_m + g, 1e-6)

        w = cp.ones((beta.shape[0], self.nu), dtype='float32')
        ramp_up = b < 2*gm_minus_g
        w = cp.where(ramp_up, cp.sin(cp.pi/4*b/gm_minus_g)**2, w)
        ramp_down = b > (scan_range - 2*gm_plus_g)
        w = cp.where(ramp_down, cp.sin(
            cp.pi/4*(scan_range-b)/gm_plus_g)**2, w)
        w = cp.where((b < 0) | (b > scan_range), 0.0, w)

        data *= w[:, cp.newaxis, :]
        return data

    def weight_and_filter(self, data, dbeta=None, parker_beta=None, parker_range=None):
        """In-place cosine (+ optional Parker) pre-weighting, ramp filtering
        along u, and (optional) per-projection quadrature-weight scaling.

        data: [ntheta, nv, nu] float32 (GPU), already dark/flat-corrected
            and negative-log'd. Returns the same buffer, filtered in place.
        dbeta: optional [ntheta] array-like (radians); the angular-sampling
            weight of each projection (e.g. trapezoidal quadrature). When
            given, the result is scaled by 0.5*dbeta so that
            FDKRec.backprojection does not need it. When omitted, the caller
            is responsible for the 0.5*dbeta normalization itself.
        parker_beta, parker_range: when a short scan needs Parker
            redundancy weighting, pass this chunk's angles relative to the
            scan start (see parker_weight) and the total scan range.
        """
        data *= self.cos_weight[cp.newaxis]
        if parker_beta is not None:
            data = self.parker_weight(data, parker_beta, parker_range)
        ntheta = data.shape[0]
        tmp = cp.empty((ntheta, self.nv, self.ne), dtype='float32')
        tmp[:, :, self.pad:self.pad+self.nu] = data
        tmp[:, :, :self.pad] = data[:, :, :1]
        tmp[:, :, self.pad+self.nu:] = data[:, :, -1:]
        tmp = cp.fft.irfft(cp.fft.rfft(tmp, axis=-1) *
                            self.wfilter, n=self.ne, axis=-1)
        data[:] = tmp[:, :, self.pad:self.pad+self.nu]
        if dbeta is not None:
            w = 0.5*cp.asarray(dbeta, dtype='float32')
            data *= w[:, cp.newaxis, cp.newaxis]
        return data

    def backprojection(self, f, data, theta, nz_off):
        """Accumulate (+=) a chunk of filtered projections into a z-chunk of
        the output volume. f is not zeroed here, so the caller can call this
        repeatedly (once per angle-chunk) to accumulate the full sum before
        reading the chunk back.

        f: [ncz, n, n] float32 (GPU) accumulator
        data: [ntheta_chunk, nv, nu] float32 (GPU), filtered + weighted,
              including the 0.5*dbeta quadrature weight (see
              weight_and_filter)
        theta: [ntheta_chunk] float32 (GPU), projection angles (radians)
        nz_off: index (in voxels) of f's first row within the full volume
        """
        ncz = f.shape[0]
        ntp = data.shape[0]
        c = 1.0/self.du
        block = (16, 16, 1)
        grid = (-(-self.n//16), -(-self.n//16), ncz)
        self._kernel(grid, block, (
            f, data, theta,
            np.int32(self.n), np.int32(ncz), np.int32(nz_off), np.int32(self.nz),
            np.int32(self.nu), np.int32(self.nv), np.int32(ntp),
            np.float32(self.voxel_size), np.float32(self.du), np.float32(self.dv),
            np.float32(self.R), np.float32(self.D),
            np.float32(self.center_u), np.float32(self.center_v),
            np.float32(c)))
