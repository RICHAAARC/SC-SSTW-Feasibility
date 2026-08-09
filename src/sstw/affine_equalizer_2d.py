"""Public-only 2D affine calibration and equalization for one AISB candidate."""

from __future__ import annotations

from dataclasses import dataclass
import math

Vector2 = tuple[float, float]


@dataclass(frozen=True)
class AffineChannel2D:
    matrix: tuple[Vector2, Vector2]
    bias: Vector2
    condition: float


def _inverse3(a: list[list[float]]) -> list[list[float]]:
    determinant = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    if abs(determinant) < 1e-12:
        raise ValueError("public calibration is singular")
    cofactors = [[(-1) ** (i + j) * (a[(i + 1) % 3][(j + 1) % 3] * a[(i + 2) % 3][(j + 2) % 3] - a[(i + 1) % 3][(j + 2) % 3] * a[(i + 2) % 3][(j + 1) % 3]) / determinant for i in range(3)] for j in range(3)]
    return cofactors


def calibrate_public_channel(pilots: tuple[Vector2, ...], observations: tuple[Vector2, ...], ridge: float, maximum_condition: float) -> AffineChannel2D:
    if len(pilots) != len(observations) or len(pilots) < 3 or ridge < 0:
        raise ValueError("public calibration inputs are invalid")
    design = [(point[0], point[1], 1.0) for point in pilots]
    normal = [[sum(row[i] * row[j] for row in design) + (ridge if i == j and i < 2 else 0.0) for j in range(3)] for i in range(3)]
    inverse = _inverse3(normal)
    coefficients = [[sum(inverse[i][j] * sum(row[j] * output[axis] for row, output in zip(design, observations, strict=True)) for j in range(3)) for i in range(3)] for axis in range(2)]
    matrix = ((coefficients[0][0], coefficients[0][1]), (coefficients[1][0], coefficients[1][1]))
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    if abs(determinant) < 1e-12:
        raise ValueError("public affine channel is rank deficient")
    frobenius = sum(value * value for row in matrix for value in row)
    discriminant = max(0.0, frobenius * frobenius - 4.0 * determinant * determinant)
    singular_max = math.sqrt((frobenius + math.sqrt(discriminant)) / 2.0)
    singular_min = abs(determinant) / singular_max
    condition = singular_max / singular_min
    if not math.isfinite(condition) or condition > maximum_condition:
        raise ValueError("public affine channel fails its condition gate")
    return AffineChannel2D(matrix, (coefficients[0][2], coefficients[1][2]), condition)


def equalize(observations: tuple[Vector2, ...], channel: AffineChannel2D) -> tuple[Vector2, ...]:
    (a, b), (c, d) = channel.matrix
    determinant = a * d - b * c
    return tuple((((d * (q[0] - channel.bias[0]) - b * (q[1] - channel.bias[1])) / determinant), ((-c * (q[0] - channel.bias[0]) + a * (q[1] - channel.bias[1])) / determinant)) for q in observations)
