import numpy as np
import skimage
from scipy.ndimage import filters, interpolation, morphology
import scipy.stats as stats


def estimate_skew(flat, bignore=0.1, maxskew=2, skewsteps=8):
    """ estimate skew angle and rotate"""
    d0, d1 = flat.shape
    o0, o1 = int(bignore * d0), int(bignore * d1)  # border ignore
    flat = np.amax(flat) - flat
    flat -= np.amin(flat)
    est = flat[o0 : d0 - o0, o1 : d1 - o1]
    ma = maxskew
    ms = int(2 * maxskew * skewsteps)
    # print(linspace(-ma,ma,ms+1))
    angle = estimate_skew_angle(est, np.linspace(-ma, ma, ms + 1))
    flat = interpolation.rotate(flat, angle, mode="constant", reshape=0)
    flat = np.amax(flat) - flat
    return flat, angle


def estimate_skew_angle(image,angles):
    estimates = []
    for a in angles:
        v = np.mean(interpolation.rotate(image, a, order=0, mode="constant"), axis=1)
        v = np.var(v)
        estimates.append((v, a))
    _,a = max(estimates)
    return a


def estimate_local_whitelevel(image, zoom=0.1, perc=80, range=20, debug=0):
    '''flatten it by estimating the local whitelevel
    zoom for page background estimation, smaller=faster, default: %(default)s
    percentage for filters, default: %(default)s
    range for filters, default: %(default)s
    '''
    m = interpolation.zoom(image, zoom)
    m = filters.percentile_filter(m, perc, size=(range, 2))
    m = filters.percentile_filter(m, perc, size=(2, range))
    m = interpolation.zoom(m, 1.0 / zoom)
    w, h = np.minimum(np.array(image.shape), np.array(m.shape))
    w_m, h_m = np.maximum(np.array(image.shape), np.array(m.shape))

    flat = np.clip(image[:w, :h]-m[:w, :h] + 1, 0, 1)
    flat = np.pad(flat, ((0, w_m - w), (0, h_m - h)), 'edge')
    flat = flat[:image.shape[0], :image.shape[1]]

    return flat


def estimate_thresholds(flat, bignore=0.1, escale=1.0, lo=5, hi=90, debug=0):
    '''# estimate low and high thresholds
    ignore this much of the border for threshold estimation, default: %(default)s
    scale for estimating a mask over the text region, default: %(default)s
    lo percentile for black estimation, default: %(default)s
    hi percentile for white estimation, default: %(default)s
    '''
    d0, d1 = flat.shape
    o0, o1 = int(bignore*d0), int(bignore*d1)
    est = flat[o0:d0-o0, o1:d1-o1]
    if escale > 0:
        # by default, we use only regions that contain
        # significant variance; this makes the percentile
        # based low and high estimates more reliable
        e = escale
        v = est-filters.gaussian_filter(est, e*20.0)
        v = filters.gaussian_filter(v**2, e*20.0)**0.5
        v = (v > 0.3*np.amax(v))
        v = morphology.binary_dilation(v, structure=np.ones((int(e*50), 1)))
        v = morphology.binary_dilation(v, structure=np.ones((1, int(e*50))))
        est = est[v]
    lo = stats.scoreatpercentile(est.ravel(), lo)
    hi = stats.scoreatpercentile(est.ravel(), hi)
    return lo, hi


def normalize_raw_image(raw):
    """ perform image normalization """
    image = raw - np.amin(raw)
    if np.amax(image) == np.amin(image):
        #print_info("# image is empty: %s" % (fname))
        return None
    image = image / np.amax(image)
    return image


def binarize(image):
    if len(image.shape) == 3:
        if image.shape[2] == 1:
            image = image[:, :, 0]
        else:
            # convert the image to greyscale
            # important to use this color space!!!!
            image = skimage.color.rgb2grey(image)
    image = normalize_raw_image(image)
    extreme = (
            (np.sum(image < 0.05) + np.sum(image > 0.95)) * 1.0 / np.prod(image.shape)
        )
    if extreme > 0.95:
        flat = image
    else:
        # if not, we need to flatten it by estimating the local whitelevel
        flat = estimate_local_whitelevel(image)
    #flat, angle = estimate_skew(flat)

    lo, hi = estimate_thresholds(flat)

    flat -= lo
    flat /= (hi-lo)
    flat = np.clip(flat, 0, 1)
    #print('Angle:'+ angle)
    return flat > 0.5