import math

import numpy as np

import util.graphics.visualisations as vis
import curves.func_based.bézier_curve as bez_c
import tensorflow as tf

def converge_shape_to_airfoil(
        points: list,
        points_num: int = 1000,
        round_digits: int = 5):
    """
    algorithm to convert an open shape like a curve into a closed one by simply connecting the first and last points;
    using DDA-like procedure
    :param points: points of the shape
    :param points_num: number of points connecting the first and last point
    :param round_digits: digits to which the calculated points should be rounded
    :return:
    """
    result = []
    start_x, start_y = points[0]
    end_x, end_y = points[-1]

    m = (end_y - start_y) / (end_x - start_x)
    cur_x = start_x
    cur_y = start_y

    step = (end_x - start_x) / points_num
    m_per_step = m * step

    for i in range(points_num):
        cur_y += m_per_step
        cur_x += step
        result.append((round(cur_x, round_digits), round(cur_y, round_digits)))

    return [*points[::-1], *result]

def converge_shape_to_mirrored_airfoil(
        points: list,
        round_digits: int = 5):
    """
    creates an XFoil-compatible airfoil from a shape by rotating it if necessary and mirroring it along the x-axis
    :param points: points to mirror
    :param round_digits: digits to which the calculated points should be rounded
    :return:
    """
    # Todo: Check whether rotation is really necessary
    # rotation checks
    #m = (points[-1][1] - points[0][1]) / (points[-1][0] - points[0][0])
    #if m != 0:
    #    points = rotate_curve(points, -1*math.degrees(math.atan(m)))

    # normalization
    points = normalize_points(points)
    # mirroring & creating airfoil
    bottom = []
    for point in points[1:]:
        bottom.append((round(point[0], round_digits), round(-1 * point[1], round_digits)))
    raw_result = [*points[::-1], *bottom]
    checked_result = []
    for element in raw_result:
        if element not in checked_result:
            checked_result.append(element)
    return checked_result
    #return checked_result, math.degrees(math.atan(m))


def _tf_resample_polyline(points: tf.Tensor, num_points: int) -> tf.Tensor:
    """
    Resampling einer 2D-Polyline auf num_points Punkte entlang der Bogenlänge (TF-only).
    points: (N, 2) float Tensor
    Rückgabe: (num_points, 2)
    """
    points = tf.convert_to_tensor(points)
    if points.shape.rank != 2 or points.shape[-1] != 2:
        raise ValueError("points muss Shape (N, 2) haben.")
    if num_points <= 1:
        raise ValueError("num_points muss > 1 sein.")

    # Segmentlängen und kumulative Länge
    diffs = points[1:] - points[:-1]                     # (N-1, 2)
    seg_len = tf.norm(diffs, axis=1)                     # (N-1,)
    cumlen = tf.concat([tf.zeros([1], dtype=seg_len.dtype),
                        tf.cumsum(seg_len)], axis=0)     # (N,)

    total_len = cumlen[-1]
    eps = tf.cast(1e-12, dtype=total_len.dtype)
    total_safe = tf.maximum(total_len, eps)

    # Zielpositionen gleichmäßig über [0, total_len]
    t = tf.linspace(tf.constant(0.0, dtype=total_safe.dtype), total_safe, num_points)  # (num_points,)

    # Für jede Zielposition Segment finden
    # idx_right: minimaler Index j, so dass cumlen[j] > t  (side="right")
    idx_right = tf.searchsorted(cumlen, t, side="right")        # (num_points,)
    idx_right = tf.clip_by_value(idx_right, 1, tf.shape(cumlen)[0] - 1)
    idx_left = idx_right - 1                                    # (num_points,)

    # Werte und Segmentgrenzen
    c0 = tf.gather(cumlen, idx_left)                            # (num_points,)
    c1 = tf.gather(cumlen, idx_right)                           # (num_points,)
    p0 = tf.gather(points, idx_left)                            # (num_points, 2)
    p1 = tf.gather(points, idx_right)                           # (num_points, 2)

    denom = tf.maximum(c1 - c0, eps)
    w = (t - c0) / denom                                        # (num_points,)
    w = tf.reshape(w, [-1, 1])                                   # (num_points, 1)

    return p0 * (1.0 - w) + p1 * w                               # (num_points, 2)


def converge_tf_shape_to_mirrored_airfoil(
    points,
    resample_req: int = 400,
    return_as: str = "tensor",
):
    """
    TF- und GradientTape-kompatible Airfoil-Erzeugung:
    1) Normalisieren wie Vorlage: x'=(x-x_min)/(x_max-x_min), y'=(y-y_min)/(x_max-x_min)
    2) Oberseite = norm[::-1]
    3) Unterseite = norm[1:] mit y -> -y
    4) Zusammenfügen (keine Repanel-/Indexsuche, vollständig differentiabel)
    5) Optionales Runden per STE, um Gradientenverlust zu vermeiden
    6) Optionale Duplikatentfernung (diskret, daher nicht differenzierbar; standardmäßig aus)

    Parameter:
      points: (N,2) float Tensor oder Liste/Tuple von (2,)-Tensors (float)
      round_digits: Nachkommastellen fürs Ausgabearrangement; None deaktiviert Runden
      return_as: "tensor" | "list"
      remove_duplicates: False empfohlen für Training (diskrete Auswahl!)
      use_ste_round: True wendet STE für Rundung an, um Gradienten zu erhalten

    Rückgabe:
      (M,2) Tensor oder Liste von (2,)-Tensors
    """
    # Eingabe -> Tensor (float, (N,2))
    if isinstance(points, (list, tuple)):
        dtype = tf.convert_to_tensor(points[0]).dtype
        if not tf.as_dtype(dtype).is_floating:
            raise TypeError("Eingabepunkte müssen Floating-Point-Dtype haben.")
        pts = tf.stack([tf.cast(p, dtype) for p in points], axis=0)
    else:
        pts = tf.convert_to_tensor(points)
        if not tf.as_dtype(pts.dtype).is_floating:
            raise TypeError("Eingabepunkte müssen Floating-Point-Dtype haben.")

    if pts.shape.rank != 2 or pts.shape[-1] != 2:
        raise ValueError("Eingabe muss Shape (N, 2) haben.")

    # Normalisierung (wie Vorlage)
    x = pts[:, 0]
    y = pts[:, 1]
    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)
    y_min = tf.reduce_min(y)

    dx = x_max - x_min
    eps = tf.cast(1e-12, dtype=pts.dtype)
    dx_safe = tf.maximum(tf.abs(dx), eps)

    x_norm = (x - x_min) / dx_safe
    y_norm = (y - y_min) / dx_safe
    norm_pts = tf.stack([x_norm, y_norm], axis=1)  # (N,2)

    # Oberseite und Unterseite (ohne Repanel, alles TF-differenzierbar)
    upper = tf.reverse(norm_pts, axis=[0])                       # (N,2)
    lower = tf.stack([norm_pts[1:, 0], -norm_pts[1:, 1]], 1)     # (N-1,2)

    airfoil = _tf_resample_polyline(tf.concat([upper, lower], axis=0), resample_req)                  # (2N-1,2)


    """
    def _ste_round(x: tf.Tensor, digits: int) -> tf.Tensor:
    
    Straight-Through-Estimator für Rundung:
    y = round(x) in Vorwärtsrichtung, Gradient ~ 1 (Identität) in Rückwärtsrichtung.
    
    if digits is None or digits < 0:
        return x
    scale = tf.cast(10 ** digits, x.dtype)
    y = tf.round(x * scale) / scale
    # STE: Vorwärts y, Gradient durch x
    return x + tf.stop_gradient(y - x)
    
 # Optionales Runden (STE, um Gradienten zu erhalten)
    if round_digits is not None:
        if use_ste_round:
            airfoil = _ste_round(airfoil, round_digits)
        else:
            # Klassisches Runden (nicht differenzierbar, Gradienten gehen verloren)
            scale = tf.cast(10 ** round_digits, dtype=airfoil.dtype)
            airfoil = tf.round(airfoil * scale) / scale
    """

    return tf.unstack(airfoil, axis=0) if return_as == "list" else airfoil

#TODO: hier sicher gehen, dass alle Gradienten safed sind & alles korrekt mit tf Operationen abläuft



def rotate_curve(points, angle: float):
    # formatting points to start at (0, 0)
    vec_x, vec_y = -points[0][0], -points[0][1]
    points_shifted = []
    for point in points:
        points_shifted.append((point[0] + vec_x, point[1] + vec_y))

    # rotation
    new_points = []
    for x, y in points_shifted:
        x_new = math.cos(math.radians(angle)) * x - math.sin(math.radians(angle)) * y
        y_new = math.sin(math.radians(angle)) * x + math.cos(math.radians(angle)) * y
        new_points.append((x_new, y_new))

    # translating back to the original starting point and returning
    return [(x-vec_x, y-vec_y) for x, y in new_points]


def normalize_points(points):
    x_min, x_max = min(points, key=lambda x: x[0])[0], max(points, key=lambda x: x[0])[0]
    y_min = min(points, key=lambda x: x[1])[1]
    d_x = x_max - x_min
    mod_list = []
    for i, point in enumerate(points):
        mod_list.append([(point[0]-x_min)/d_x, (point[1]-y_min)/d_x])
    return mod_list

def normalize_tf_points(points, return_as="tensor"):
    """
    # Credit: ChatGPT-5
    Normalisiert 2D-Punkte mit TensorFlow-Operationen:
    x' = (x - x_min) / (x_max - x_min)
    y' = (y - y_min) / (x_max - x_min)
    (Skalierung beider Achsen über die x-Spannweite wie in deinem Beispiel.)
    Parameter:
      points:
        - tf.Tensor mit Shape (N, 2) oder
        - Liste/Tuple aus tf.Tensor/tf.Variable mit Shape (2,) pro Punkt
      return_as: "tensor" | "list"
        - "tensor": Rückgabe als tf.Tensor (N, 2)
        - "list": Rückgabe als Liste von tf.Tensor der Länge N (je (2,))
    Rückgabe:
      Normalisierte Punkte (Tensor oder Liste), Gradienten fließen zurück.
    """
    # In Tensor zusammenführen, ohne dtype zu verändern
    if isinstance(points, (list, tuple)):
        # Dtype von erstem Element übernehmen (muss float sein für Gradienten)
        dtype = points[0].dtype
        # Sicherheits-Check: Gradienten nur für Floating-Types sinnvoll
        if not tf.as_dtype(dtype).is_floating:
            raise TypeError("normalize_tf_points erwartet Floating-Point-Dtype für Gradient-Unterstützung.")
        pts = tf.stack(points, axis=0)  # Shape (N, 2)
    else:
        pts = tf.convert_to_tensor(points)
        if not tf.as_dtype(pts.dtype).is_floating:
            raise TypeError("normalize_tf_points erwartet Floating-Point-Dtype für Gradient-Unterstützung.")

    # Erwartet 2D-Punkte
    if pts.shape.rank != 2 or pts.shape[-1] != 2:
        raise ValueError("Eingabe muss Shape (N, 2) haben.")

    # Min/Max und Skalenfaktoren (nur TF-Operationen)
    x = pts[:, 0]
    y = pts[:, 1]

    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)
    y_min = tf.reduce_min(y)

    dx = x_max - x_min
    eps = tf.cast(1e-12, dtype=pts.dtype)
    dx_safe = tf.where(tf.abs(dx) > eps, dx, eps)  # Numerisch stabil

    x_norm = (x - x_min) / dx_safe
    y_norm = (y - y_min) / dx_safe

    out = tf.stack([x_norm, y_norm], axis=1)  # (N, 2), gleicher dtype wie Eingabe

    if return_as == "list":
        # Struktur wie Liste von Punkten (je (2,)) – Gradienten bleiben intakt
        out_list = tf.unstack(out, axis=0)
        return out_list
    return out


def __demonstration_correct_airfoil_conversion__():
    cont_points = [(1, 1), (2, 7), (3, 5), (5, 3)]
    bez_curve = bez_c.bezier_curve(cont_points, 100)

    vis.visualize_curve(bez_curve, cont_points, True)
    rotated_bez_curve = converge_shape_to_mirrored_airfoil(bez_curve, 5)[0]
    vis.visualize_curve(rotated_bez_curve, cont_points, True)