#!/usr/bin/env python
# -*- coding: utf-8 -*-

# *************************************************************************** #
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

import os
import glob
import signal
import time

import numpy as np
import h5py
import cupy as cp
import tifffile

from tomocupy import utils
from tomocupy import logging
from tomocupy.processing import proc_functions
from tomocupy.processing import remove_stripe
from tomocupy.reconstruction import fdk
from tomocupy.dataio import writer as dio_writer
from tomocupy.global_vars import args, params

__author__ = "Tomocupy contributors"
__docformat__ = 'restructuredtext en'
__all__ = ['GPURecFDK', ]

log = logging.getLogger(__name__)

_TWO_PI = 2*np.pi


class GPURecFDK():
    """Cone-beam FDK reconstruction for lab/micro-CT (flat-panel detector,
    circular trajectory), on GPU.

    This is a self-contained pipeline independent of GPURec/GPURecSteps: the
    parallel-beam sinogram chunking those use (equal-size slices in z, reused
    across every projection angle) does not apply here, since a cone-beam ray
    through any output z-chunk can come from anywhere on the detector.
    Instead this runs three passes over the data:

      1. correction pass: dark/flat correction and -log, chunked by
         projection angle, with disk reads overlapped (via a background
         thread) with GPU compute.
      2. ring removal (optional): unlike the rest of the filtering, stripe
         removal needs the full angular range at once, so it runs as its own
         pass chunked by detector row instead of by angle.
      3. weighting + filtering pass: FDK cosine weighting, the Parker
         redundancy weight for scans under 360 degrees, ramp filtering, and
         the per-projection quadrature weight -- chunked by projection angle.

    All three passes read/write a single float32 array the size of the input
    data, kept in ordinary (unpinned) host RAM between passes (never
    re-read from disk). The final pass is:

      4. backprojection pass: chunked by output z-slice; for each chunk, the
         filtered sinogram is streamed to the GPU in angle chunks and
         accumulated with the FDK backprojection kernel
         (tomocupy.reconstruction.fdk). Each finished chunk is handed to a
         background writer thread while the GPU moves on to the next chunk.
    """

    def __init__(self):
        signal.signal(signal.SIGINT, utils.signal_handler)
        signal.signal(signal.SIGTERM, utils.signal_handler)
        cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)

        if args.dark_file_name is None:
            args.dark_file_name = args.file_name
        if args.flat_file_name is None:
            args.flat_file_name = args.file_name

        # Settings implied by this reconstruction path, needed by the reused
        # dark/flat + stripe-removal + zarr-writing code elsewhere in tomocupy:
        args.dtype = 'float32'
        args.flat_linear = 'False'
        args.beam_hardening_method = 'none'
        args.large_data = False

        self._init_geometry()
        self.cl_proc_func = proc_functions.ProcFunctions()
        self.fdk = fdk.FDKRec(
            self.nu, self.nv, self.n, self.nz,
            args.source_origin_dist, args.source_detector_dist,
            self.du, self.dv, self.voxel_size,
            args.center_u, args.center_v, args.fdk_filter)

        if self.is_short_scan:
            gamma_m = self.fdk.fan_angle(gamma_max_only=True)
            min_range = np.pi+2*gamma_m
            if self.parker_range < min_range:
                log.warning(
                    f'Scan range {np.degrees(self.parker_range):.1f} deg is below the '
                    f'Parker minimum ({np.degrees(min_range):.1f} deg) for this geometry: '
                    'the reconstruction will be based on incomplete data')

        self._init_output()

    def _init_geometry(self):
        if args.source_origin_dist <= 0 or args.source_detector_dist <= 0 or args.detector_pixel_size <= 0:
            raise RuntimeError(
                'source-origin-dist, source-detector-dist and detector-pixel-size '
                'are required (and must be positive) for fdk reconstruction')
        if args.source_detector_dist <= args.source_origin_dist:
            raise RuntimeError(
                'source-detector-dist must be larger than source-origin-dist')

        with h5py.File(args.file_name, 'r') as fid:
            nproji, nzi, nii = fid['/exchange/data'].shape
            if '/exchange/theta' in fid:
                theta = fid['/exchange/theta'][:].astype('float64')/180*np.pi
            else:
                theta = np.linspace(
                    0, args.full_rotation_deg/180*np.pi, nproji, endpoint=False)

        if args.end_row == -1:
            args.end_row = nzi
        if args.end_column == -1:
            args.end_column = nii
        if args.end_proj == -1:
            args.end_proj = nproji

        theta = theta[args.start_proj:args.end_proj]
        nproj = len(theta)
        if nproj < 2:
            raise RuntimeError('need at least 2 projections for fdk reconstruction')

        # downsample() halves each axis per binning level by pairing up
        # neighboring pixels, so the cropped row/column ranges need to be
        # exact multiples of 2**binning
        step = 2**args.binning
        nsh = (args.end_column-args.start_column) % step
        if nsh != 0:
            args.end_column -= nsh
            log.warning(
                f'Decreasing detector width by {nsh} pixel(s) to operate with multiples of {step}')
        nsh = (args.end_row-args.start_row) % step
        if nsh != 0:
            args.end_row -= nsh
            log.warning(
                f'Decreasing detector height by {nsh} pixel(s) to operate with multiples of {step}')

        nu = (args.end_column - args.start_column)//step
        nv = (args.end_row - args.start_row)//step
        if nu < 2 or nv < 2:
            raise RuntimeError('cropped/binned detector size is too small')

        du = args.detector_pixel_size*2**args.binning
        dv = du
        mag = args.source_origin_dist/args.source_detector_dist
        voxel_size = args.volume_pixel_size if args.volume_pixel_size > 0 else du*mag

        n = args.volume_size if args.volume_size > 0 else nu
        nz = args.volume_height if args.volume_height > 0 else nv

        self.nu, self.nv, self.n, self.nz = nu, nv, n, nz
        self.du, self.dv, self.voxel_size = du, dv, voxel_size
        self.nproj = nproj
        self.theta = theta.astype('float32')

        self._init_angular_weights(theta)

        self.ncproj = min(args.nproj_per_chunk, nproj)
        self.ntchunk = int(np.ceil(nproj/self.ncproj))
        self.ltchunk = np.minimum(
            self.ncproj, nproj-np.arange(self.ntchunk)*self.ncproj)

        self.ncz = min(args.nz_per_chunk, nz)
        self.nzchunk = int(np.ceil(nz/self.ncz))
        self.lzchunk = np.minimum(
            self.ncz, nz-np.arange(self.nzchunk)*self.ncz)

        log.info(
            f'fdk geometry: detector {nu}x{nv}, volume {n}x{n}x{nz}, '
            f'{nproj} projections, voxel size {voxel_size:.6g}, '
            f'magnification {1/mag:.4g}')

    def _init_angular_weights(self, theta):
        """Per-projection quadrature weight (exact for non-uniform angle
        spacing, periodic for a full rotation) and whether/how to apply the
        Parker short-scan redundancy weight.
        """
        sign = 1.0 if theta[-1] >= theta[0] else -1.0
        span = abs(float(theta[-1]-theta[0]))
        nproj = len(theta)
        avg_step = span/(nproj-1)
        is_full_scan = abs(span+avg_step-_TWO_PI) < 0.5*avg_step

        dtheta = np.empty(nproj, dtype='float64')
        if is_full_scan:
            nxt = np.roll(theta, -1).copy()
            nxt[-1] = theta[0] + sign*_TWO_PI
            prv = np.roll(theta, 1).copy()
            prv[0] = theta[-1] - sign*_TWO_PI
            dtheta = np.abs(nxt-prv)/2
        else:
            dtheta[1:-1] = np.abs(theta[2:]-theta[:-2])/2
            dtheta[0] = np.abs(theta[1]-theta[0])
            dtheta[-1] = np.abs(theta[-1]-theta[-2])
        self.dbeta_per_proj = dtheta.astype('float32')

        want_parker = (args.short_scan_weighting == 'parker' or
                      (args.short_scan_weighting == 'auto' and not is_full_scan))
        self.is_short_scan = want_parker
        if want_parker:
            self.parker_beta = (theta-theta[0]).astype('float32')*sign
            self.parker_range = span
            log.warning(
                f'Scan range {np.degrees(span):.1f} deg < 360: applying Parker '
                'short-scan redundancy weighting')
        else:
            self.parker_beta = None
            self.parker_range = None

    def _init_output(self):
        if args.out_path_name is None:
            base = os.path.basename(str(args.file_name))
            if '.' in base:
                base = base.rsplit('.', 1)[0]
            fnameout = os.path.dirname(str(args.file_name))+'_rec/'+base+'_rec'
        else:
            fnameout = str(args.out_path_name)

        self.h5file = None
        self.zarr_array = None
        if args.save_format == 'tiff':
            if not os.path.exists(fnameout):
                os.makedirs(fnameout)
            if args.clear_folder == 'True':
                for f in glob.glob(os.path.join(fnameout, '*')):
                    os.remove(f)
            self.fnameout = os.path.join(fnameout, 'recon')
        elif args.save_format == 'h5nolinks':
            parent = os.path.dirname(fnameout)
            if parent and not os.path.exists(parent):
                os.makedirs(parent)
            h5name = fnameout+'.h5'
            h5w = h5py.File(h5name, 'w')
            dset = h5w.create_dataset(
                '/exchange/data', shape=(self.nz, self.n, self.n), dtype='float32')
            dset.attrs['axes'] = 'z:y:x'
            dset.attrs['description'] = 'ReconData'
            self.h5file = h5w
            self.h5dset = dset
            self.fnameout = h5name
        elif args.save_format == 'h5':
            self.h5_parts_dir = f'{fnameout}_parts'
            if not os.path.exists(self.h5_parts_dir):
                os.makedirs(self.h5_parts_dir)
            self.fnameout = fnameout+'.h5'
        elif args.save_format == 'zarr':
            self.fnameout = fnameout+'.zarr'
            self.zarr_output_path = self.fnameout
            dio_writer.clean_zarr(self.zarr_output_path)
        log.info(f'Output: {self.fnameout}')

    def _init_zarr(self):
        shape = (self.nz, self.n, self.n)
        chunks = [int(c.strip()) for c in args.zarr_chunk.split(',')]
        max_levels = (lambda X, Y: (lambda r: (int(r).bit_length()-1)
                      if r != 0 else (X.bit_length()-1))(int(X) % int(Y)))
        levels = min(max_levels(self.nz, self.ncz), 6)
        scale_factors = [float(self.voxel_size)*(i+1) for i in range(levels)]
        self.zarr_array, datasets = dio_writer.initialize_zarr(
            output_path=self.zarr_output_path, base_shape=shape, chunks=chunks,
            dtype='float32', num_levels=levels, scale_factors=scale_factors,
            compression=args.zarr_compression)
        dio_writer.fill_zarr_meta(self.zarr_array, datasets, self.zarr_output_path, args)

    def _write_chunk(self, rec, st, end, k):
        if args.save_format == 'tiff':
            for kk in range(end-st):
                tifffile.imwrite(f'{self.fnameout}_{st+kk:05d}.tiff', rec[kk])
        elif args.save_format == 'h5nolinks':
            self.h5dset[st:end] = rec[:end-st]
        elif args.save_format == 'h5':
            filename = f'{self.h5_parts_dir}/p{k:04d}.h5'
            with h5py.File(filename, 'w') as fid:
                fid.create_dataset('/exchange/data', data=rec[:end-st],
                                   chunks=(1, self.n, self.n))
        elif args.save_format == 'zarr':
            if self.zarr_array is None:
                self._init_zarr()
            dio_writer.write_zarr_chunk(self.zarr_array, rec[:end-st], st, end)

    def _finalize_h5_virtual(self):
        layout = h5py.VirtualLayout(
            shape=(self.nz, self.n, self.n), dtype='float32')
        for k in range(self.nzchunk):
            filename = f'{self.h5_parts_dir}/p{k:04d}.h5'
            st = k*self.ncz
            vsource = h5py.VirtualSource(
                filename, '/exchange/data', shape=(self.lzchunk[k], self.n, self.n),
                dtype='float32')
            layout[st:st+self.lzchunk[k]] = vsource
        with h5py.File(self.fnameout, 'w') as hf:
            dset = hf.create_virtual_dataset('/exchange/data', layout)
            dset.attrs['axes'] = 'z:y:x'
            dset.attrs['description'] = 'ReconData'

    def _read_proj_chunk(self, st, end):
        sr, er = args.start_row, args.end_row
        sc, ec = args.start_column, args.end_column
        with h5py.File(args.file_name, 'r') as fid:
            d = fid['/exchange/data'][args.start_proj+st:args.start_proj+end, sr:er, sc:ec]
        return utils.downsample(d.astype('float32', copy=False), args.binning)

    def _read_flat_dark(self):
        sr, er = args.start_row, args.end_row
        sc, ec = args.start_column, args.end_column
        with h5py.File(args.dark_file_name, 'r') as fid:
            dark = fid['/exchange/data_dark'][:, sr:er, sc:ec]
        with h5py.File(args.flat_file_name, 'r') as fid:
            flat = fid['/exchange/data_white'][:, sr:er, sc:ec]
        return (utils.downsample(flat.astype('float32', copy=False), args.binning),
                utils.downsample(dark.astype('float32', copy=False), args.binning))

    def _correction_pass(self):
        """Dark/flat correction and -log, chunked by projection angle with
        disk reads overlapped (background thread) with GPU compute. Returns
        the corrected sinogram as a single float32 array in host RAM, shaped
        [nproj, nv, nu].
        """
        nu, nv = self.nu, self.nv
        ncproj, ntchunk, ltchunk = self.ncproj, self.ntchunk, self.ltchunk

        flat, dark = self._read_flat_dark()
        flat_gpu = cp.asarray(flat)
        dark_gpu = cp.asarray(dark)

        corrected = np.empty((self.nproj, nv, nu), dtype='float32')
        host_buf = [np.empty((ncproj, nv, nu), dtype='float32') for _ in range(2)]
        read_thread = utils.WRThread()

        def _read(buf, st, end):
            buf[:end-st] = self._read_proj_chunk(st, end)

        _read(host_buf[0], 0, ltchunk[0])

        for k in range(ntchunk):
            utils.printProgressBar(k, ntchunk, 0, length=40)
            st = k*ncproj
            end = st+ltchunk[k]
            cur = host_buf[k % 2]

            if k+1 < ntchunk:
                nst = (k+1)*ncproj
                nend = nst+ltchunk[k+1]
                read_thread.run(_read, (host_buf[(k+1) % 2], nst, nend))

            data_gpu = cp.asarray(cur[:end-st])
            data_gpu = self.cl_proc_func.proc_sino(data_gpu, dark_gpu, flat_gpu)
            if args.minus_log == 'True':
                data_gpu = self.cl_proc_func.minus_log(data_gpu)
            corrected[st:end] = cp.asnumpy(data_gpu)

            if k+1 < ntchunk:
                read_thread.join()

        return corrected

    def _remove_stripe_full(self, corrected):
        """Ring removal over the full angular range, chunked by detector
        row (the ring-removal statistics need every projection at once,
        unlike the rest of the filtering pass).
        """
        nv = self.nv
        nrc = min(args.stripe_rows_per_chunk, nv)
        nrchunk = int(np.ceil(nv/nrc))
        for kr in range(nrchunk):
            utils.printProgressBar(kr, nrchunk, 0, length=40)
            rst = kr*nrc
            rend = min(rst+nrc, nv)
            chunk_gpu = cp.asarray(corrected[:, rst:rend, :])
            if args.remove_stripe_method == 'fw':
                chunk_gpu = remove_stripe.remove_stripe_fw(
                    chunk_gpu, args.fw_sigma, args.fw_filter, args.fw_level)
            elif args.remove_stripe_method == 'ti':
                chunk_gpu = remove_stripe.remove_stripe_ti(
                    chunk_gpu, args.ti_beta, args.ti_mask)
            elif args.remove_stripe_method == 'vo-all':
                chunk_gpu = remove_stripe.remove_all_stripe(
                    chunk_gpu, args.vo_all_snr, args.vo_all_la_size,
                    args.vo_all_sm_size, args.vo_all_dim)
            corrected[:, rst:rend, :] = cp.asnumpy(chunk_gpu)

    def _weight_filter_pass(self, corrected):
        """FDK cosine weighting, optional Parker weighting, ramp filtering
        and per-projection quadrature-weight scaling, chunked by projection
        angle. Writes the result back into `corrected` in place (renamed
        conceptually to "filtered") to avoid doubling host memory use.
        """
        ncproj, ntchunk, ltchunk = self.ncproj, self.ntchunk, self.ltchunk
        for k in range(ntchunk):
            utils.printProgressBar(k, ntchunk, 0, length=40)
            st = k*ncproj
            end = st+ltchunk[k]
            data_gpu = cp.asarray(corrected[st:end])
            dbeta_chunk = cp.asarray(self.dbeta_per_proj[st:end])
            if self.is_short_scan:
                parker_beta_chunk = cp.asarray(self.parker_beta[st:end])
                data_gpu = self.fdk.weight_and_filter(
                    data_gpu, dbeta=dbeta_chunk,
                    parker_beta=parker_beta_chunk, parker_range=self.parker_range)
            else:
                data_gpu = self.fdk.weight_and_filter(data_gpu, dbeta=dbeta_chunk)
            corrected[st:end] = cp.asnumpy(data_gpu)
        return corrected

    def _filter_pass(self):
        log.info('FDK: dark/flat correction')
        corrected = self._correction_pass()

        if args.remove_stripe_method != 'none':
            log.info('FDK: ring removal (full angular range)')
            self._remove_stripe_full(corrected)

        log.info('FDK: cosine/Parker weighting + ramp filtering')
        return self._weight_filter_pass(corrected)

    def _backprojection_pass(self, filtered):
        n, ncz, nzchunk, lzchunk = self.n, self.ncz, self.nzchunk, self.lzchunk
        ncproj, ntchunk, ltchunk = self.ncproj, self.ntchunk, self.ltchunk

        theta_gpu = cp.asarray(self.theta)
        theta_chunks = [theta_gpu[k*ncproj:k*ncproj+ltchunk[k]]
                        for k in range(ntchunk)]

        proj_gpu = cp.zeros((ncproj, self.nv, self.nu), dtype='float32')
        rec_gpu = cp.zeros((ncz, n, n), dtype='float32')
        rec_pinned = utils.pinned_array(
            np.zeros((args.max_write_threads, ncz, n, n), dtype='float32'))

        write_threads = [utils.WRThread() for _ in range(args.max_write_threads)]

        for kz in range(nzchunk):
            utils.printProgressBar(kz, nzchunk, 0, length=40)
            zst = kz*ncz
            zend = zst+lzchunk[kz]
            f = rec_gpu[:zend-zst]
            f[:] = 0
            for kt in range(ntchunk):
                nst = kt*ncproj
                nend = nst+ltchunk[kt]
                proj_gpu[:nend-nst].set(filtered[nst:nend])
                self.fdk.backprojection(
                    f, proj_gpu[:nend-nst], theta_chunks[kt], zst)

            ithread = utils.find_free_thread(write_threads)
            f.get(out=rec_pinned[ithread, :zend-zst])
            write_threads[ithread].run(
                self._write_chunk, (rec_pinned[ithread, :zend-zst], zst, zend, kz))

        for t in write_threads:
            t.join()

    def recon_all(self):
        """Full cone-beam FDK reconstruction."""
        t0 = time.time()
        filtered = self._filter_pass()
        t1 = time.time()
        log.info(f'Filtering done in {t1-t0:.1f}s')

        log.info('FDK: backprojection')
        self._backprojection_pass(filtered)
        t2 = time.time()
        log.info(f'Backprojection done in {t2-t1:.1f}s')

        if args.save_format == 'h5':
            self._finalize_h5_virtual()
        if self.h5file is not None:
            self.h5file.close()
