import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class wienerModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.reg_param = torch.nn.Parameter(0.1*torch.ones(1))

    def forward(self, meas, psf):
        meas_fft = torch.fft.rfft2(meas)
        psf_fft = torch.fft.rfft2(psf, meas.size()[-2:])

        impr = self.laplacian(2)
        impr = impr[np.newaxis, np.newaxis, :, :]
        reg = torch.fft.rfft2(torch.from_numpy(impr), meas.size()[-2:]) # Regularization

        wiener_filter = torch.conj(psf_fft)/(torch.abs(psf_fft)**2 + self.reg_param*reg**2)
        deconvolved = torch.fft.irfft2(wiener_filter*meas_fft)

        return deconvolved[:,:,:320,:320]

    def laplacian(self, ndim):
        impr = np.zeros([3] * ndim)
        for dim in range(ndim):
            idx = tuple([slice(1, 2)] * dim +
                        [slice(None)] +
                        [slice(1, 2)] * (ndim - dim - 1))
            impr[idx] = np.array([-1.0,
                                0.0,
                                -1.0]).reshape([-1 if i == dim else 1
                                                for i in range(ndim)])
        impr[(slice(1, 2), ) * ndim] = 2.0 * ndim
        return impr