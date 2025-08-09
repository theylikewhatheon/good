"""Vector utilities for the Nexto bot."""

import math


class Vec3:
    """A minimal vector class to mirror the logic from the Kotlin bot."""
    
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    @staticmethod
    def from_rlbot_vector(vec):
        """Create Vec3 from RLBot vector."""
        return Vec3(vec.x, vec.y, vec.z) if vec is not None else Vec3()

    def __add__(self, other):
        """Add two vectors."""
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        """Subtract two vectors."""
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        """Multiply vector by scalar."""
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def magnitude(self):
        """Calculate vector magnitude."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self):
        """Return normalized vector."""
        mag = self.magnitude()
        return self if mag == 0 else Vec3(self.x / mag, self.y / mag, self.z / mag)

    def distance(self, other):
        """Calculate distance to another vector."""
        return (self - other).magnitude()

    def dot(self, other):
        """Calculate dot product with another vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __repr__(self):
        """String representation."""
        return f"Vec3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"