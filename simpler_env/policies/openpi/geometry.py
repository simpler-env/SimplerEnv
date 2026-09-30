"""
Mostly copied from transforms3d library

"""

import math

import numpy as np

_FLOAT_EPS = np.finfo(np.float64).eps

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# For testing whether a number is close to zero
_EPS4 = np.finfo(float).eps * 4.0


def mat2euler(mat, axes="sxyz"):
    """Return Euler angles from rotation matrix for specified axis sequence.

    Note that many Euler angle triplets can describe one matrix.

    Parameters
    ----------
    mat : array-like shape (3, 3) or (4, 4)
        Rotation matrix or affine.
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).

    Examples
    --------
    >>> R0 = euler2mat(1, 2, 3, 'syxz')
    >>> al, be, ga = mat2euler(R0, 'syxz')
    >>> R1 = euler2mat(al, be, ga, 'syxz')
    >>> np.allclose(R0, R1)
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.array(mat, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS4:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS4:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def quat2mat(q):
    """Calculate rotation matrix corresponding to quaternion

    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows quaternions that
    have not been normalized.

    References
    ----------
    Algorithm from http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    """
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < _FLOAT_EPS:
        return np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    return np.array(
        [
            [1.0 - (yY + zZ), xY - wZ, xZ + wY],
            [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
            [xZ - wY, yZ + wX, 1.0 - (xX + yY)],
        ]
    )


# Checks if a matrix is a valid rotation matrix.
def isrotation(
    R: np.ndarray,
    thresh=1e-6,
) -> bool:
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    iden = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(iden - shouldBeIdentity)
    return n < thresh


def euler2mat(ai, aj, ak, axes="sxyz"):
    """Return rotation matrix from Euler angles and axis sequence.

    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    mat : array (3, 3)
        Rotation matrix or affine.

    Examples
    --------
    >>> R = euler2mat(1, 2, 3, 'syxz')
    >>> np.allclose(np.sum(R[0]), -1.34786452)
    True
    >>> R = euler2mat(1, 2, 3, (0, 1, 0, 1))
    >>> np.allclose(np.sum(R[0]), -0.383436184)
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    M = np.eye(3)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * ss + cc
        M[j, k] = -cj * cs - sc
        M[k, i] = -sj * ck
        M[k, j] = cj * sc + cs
        M[k, k] = cj * cc - ss
    else:
        M[i, i] = cj * ck
        M[i, j] = sj * sc - cs
        M[i, k] = sj * cc + ss
        M[j, i] = cj * sk
        M[j, j] = sj * ss + cc
        M[j, k] = sj * cs - sc
        M[k, i] = -sj
        M[k, j] = cj * si
        M[k, k] = cj * ci
    return M


def euler2axangle(ai, aj, ak, axes="sxyz"):
    """Return angle, axis corresponding to Euler angles, axis specification

    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    vector : array shape (3,)
       axis around which rotation occurs
    theta : scalar
       angle of rotation

    Examples
    --------
    >>> vec, theta = euler2axangle(0, 1.5, 0, 'szyx')
    >>> np.allclose(vec, [0, 1, 0])
    True
    >>> theta
    1.5
    """
    return quat2axangle(euler2quat(ai, aj, ak, axes))


def euler2quat(ai, aj, ak, axes="sxyz"):
    """Return `quaternion` from Euler angles and axis sequence `axes`

    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    quat : array shape (4,)
       Quaternion in w, x, y z (real, then vector) format

    Examples
    --------
    >>> q = euler2quat(1, 2, 3, 'ryxz')
    >>> np.allclose(q, [0.435953, 0.310622, -0.718287, 0.444435])
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = _NEXT_AXIS[i + parity - 1] + 1
    k = _NEXT_AXIS[i - parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai = ai / 2.0
    aj = aj / 2.0
    ak = ak / 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    q = np.empty((4,))
    if repetition:
        q[0] = cj * (cc - ss)
        q[i] = cj * (cs + sc)
        q[j] = sj * (cc + ss)
        q[k] = sj * (cs - sc)
    else:
        q[0] = cj * cc + sj * ss
        q[i] = cj * sc - sj * cs
        q[j] = cj * ss + sj * cc
        q[k] = cj * cs - sj * sc
    if parity:
        q[j] *= -1.0

    return q


def quat2axangle(quat, identity_thresh=None):
    """Convert quaternion to rotation of angle around axis

    Parameters
    ----------
    quat : 4 element sequence
       w, x, y, z forming quaternion.
    identity_thresh : None or scalar, optional
       Threshold below which the norm of the vector part of the quaternion (x,
       y, z) is deemed to be 0, leading to the identity rotation.  None (the
       default) leads to a threshold estimated based on the precision of the
       input.

    Returns
    -------
    theta : scalar
       angle of rotation.
    vector : array shape (3,)
       axis around which rotation occurs.

    Examples
    --------
    >>> vec, theta = quat2axangle([0, 1, 0, 0])
    >>> vec
    array([1., 0., 0.])
    >>> np.allclose(theta, np.pi)
    True

    If this is an identity rotation, we return a zero angle and an arbitrary
    vector:

    >>> quat2axangle([1, 0, 0, 0])
    (array([1., 0., 0.]), 0.0)

    If any of the quaternion values are not finite, we return a NaN in the
    angle, and an arbitrary vector:

    >>> quat2axangle([1, np.inf, 0, 0])
    (array([1., 0., 0.]), nan)

    Notes
    -----
    A quaternion for which x, y, z are all equal to 0, is an identity rotation.
    In this case we return a 0 angle and an arbitrary vector, here [1, 0, 0].

    The algorithm allows for quaternions that have not been normalized.
    """
    quat = np.asarray(quat)
    Nq = np.sum(quat**2)
    if not np.isfinite(Nq):
        return np.array([1.0, 0, 0]), float("nan")
    if identity_thresh is None:
        try:
            identity_thresh = np.finfo(Nq.type).eps * 3
        except (AttributeError, ValueError):  # Not a numpy type or not float
            identity_thresh = _FLOAT_EPS * 3
    if Nq < _FLOAT_EPS**2:  # Results unreliable after normalization
        return np.array([1.0, 0, 0]), 0.0
    if Nq != 1:  # Normalize if not normalized
        s = math.sqrt(Nq)
        quat = quat / s
    xyz = quat[1:]
    len2 = np.sum(xyz**2)
    if len2 < identity_thresh**2:
        # if vec is nearly 0,0,0, this is an identity rotation
        return np.array([1.0, 0, 0]), 0.0
    # Make sure w is not slightly above 1 or below -1
    theta = 2 * math.acos(max(min(quat[0], 1), -1))
    return xyz / math.sqrt(len2), theta


def quat2euler(quaternion, axes="sxyz"):
    """Euler angles from `quaternion` for specified axis sequence `axes`

    Parameters
    ----------
    q : 4 element sequence
       w, x, y, z of quaternion
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).

    Examples
    --------
    >>> angles = quat2euler([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(angles, [0.123, 0, 0])
    True
    """
    return mat2euler(quat2mat(quaternion), axes)