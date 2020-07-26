import numpy as np

EPSILON = 1E-10


def calc_contrast(brighter, darker, method='weber', mode='elementwise'):
    """
    Contrast metric calculation. Now supports Weber and Michelson metric.
    :param brighter: ndarray of the brighter RoI
    :param darker: ndarray of the darker RoI
    :param method: contrast metric: 'weber' | 'michelson'
    :param mode: 'pairwise' | 'elementwise'. If pairwise, contrasts will
        be calculated for all possible pixel-pairs from the brighter and
        darker RoIs. If elementwise, contrasts will be calculated for
        pairs from the sample pixel locations.
    :return:
    """
    if brighter.ndim > 1:
        brighter = brighter.ravel()
    if darker.ndim > 1:
        darker = darker.ravel()

    length_diff = brighter.size - darker.size
    if length_diff < 0:
        brighter = np.hstack([brighter, np.random.choice(brighter, size=length_diff)])
    if length_diff > 0:
        darker = np.hstack([darker, np.random.choice(darker, size=length_diff)])

    # auto-broadcasting achieves better performance than for-loop
    if mode == 'pairwise':
        brighter = brighter[:, None]
        darker = darker[None, :]

    if method.lower() == 'weber':
        contrast = brighter / (darker + EPSILON)
    elif method.lower() == 'michelson':
        contrast = (brighter - darker) / (brighter + darker + EPSILON)
    else:
        raise ValueError('{} is not a valid contrast metric.'.format(method))

    return contrast.ravel() if mode == 'pairwise' else contrast
