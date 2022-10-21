from cgi import test
import math

class Vector(object):
    
    CANNOT_NORMALIZE_VECTOR = "Zero vector normalization error"
    NO_UNIQUE_PARALLEL_COMPONENT = "There is no parallel component"
    NO_UNIQUE_ORTHOGONAL_COMPONENT = "There is no orthogonal component"
    
    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple(coordinates)
            self.dimension = len(coordinates)

        except ValueError:
            raise ValueError('The coordinates must be nonempty')

        except TypeError:
            raise TypeError('The coordinates must be an iterable')
    
    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)

    def __eq__(self, v):
        return self.coordinates == v.coordinates
    
    def plus(self, v):
        new_coordinates = []
        for i in range(len(self.coordinates)):
            new_coordinates.append(self.coordinates[i] + v.coordinates[i])
        return Vector(new_coordinates)
    
    def minus(self, v):
        new_coordinates = []
        for i in range(len(self.coordinates)):
            new_coordinates.append(self.coordinates[i] - v.coordinates[i])
        return Vector(new_coordinates)
    
    def times_scalar(self, t):
        new_coordinates = []
        for i in range(len(self.coordinates)):
            new_coordinates.append(self.coordinates[i] * t)
        return Vector(new_coordinates)
    
    def magnitude(self):
        new_coordinates = []
        for i in range(len(self.coordinates)):
            new_coordinates.append(self.coordinates[i]**2)
        return math.sqrt(sum(new_coordinates))

    def normalized(self):
        try:
            t = self.magnitude()
            return self.times_scalar(1/t)
        except ZeroDivisionError:
            raise Exception(self.CANNOT_NORMALIZE_VECTOR)
    
    def dot_product(self, v):
        new_coordinates = []
        for i in range(len(self.coordinates)):
            new_coordinates.append(self.coordinates[i] * v.coordinates[i])
        return sum(new_coordinates)
    
    def get_angle(self, v, in_degrees=False):
        try:
            u1 = self.normalized()
            v1 = v.normalized()
            angle = math.acos(u1.dot_product(v1))
            if in_degrees:
                degrees_in_radians = 180 / math.pi
                return angle * degrees_in_radians
            else:
                return angle
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_VECTOR:
                raise Exception(self.CANNOT_NORMALIZE_VECTOR)
            else:
                raise (e)
    
    def is_orthogonal(self, v, limit=1e-10):
        return abs(self.dot_product(v)) < limit
    
    def is_zero(self, limit=1e-10):
        return self.magnitude() < limit
    
    def is_parallel(self, v):
        return (self.is_zero() or v.is_zero() or self.get_angle(v) == 0 or self.get_angle(v) == math.pi)

    def component_orthogonal(self, basis):
        try:
            projection = self.component_parallel(basis)
            return self.minus(projection)
        except Exception as e:
            if str(e) == self.NO_UNIQUE_PARALLEL_COMPONENT:
                raise Exception(self.NO_UNIQUE_ORTHOGONAL_COMPONENT)
            else:
                raise (e)
            
    def component_parallel(self, basis):
        try:
            u = basis.normalized()
            weight = self.dot_product(u)
            return u.times_scalar(weight)
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_VECTOR:
                raise Exception(self.NO_UNIQUE_PARALLEL_COMPONENT)
            else:
                raise (e)
    
    def cross_product(self, v):
        try:
            Ux, Uy, Uz = self.coordinates
            Vx, Vy, Vz = v.coordinates
            new_coordinates = []
            new_coordinates.append(Uy*Vz - Vy*Uz)
            new_coordinates.append(-(Ux*Vz - Vx*Uz))
            new_coordinates.append(Ux*Vy - Vx*Uy)
            return Vector(new_coordinates)
        except ValueError as e:
            raise e
    
    def area_of_parallelogram(self, v):
        cross_product = self.cross_product(v)
        return cross_product.magnitude()
    
    def area_of_triangle(self, v):
        return self.area_of_parallelogram(v) / 2.0
   
# def main():

#     v = Vector([8.462, 7.893, -8.187])
#     u = Vector([6.984, -5.975, 4.778])
#     print(v.cross_product(u))
#     v = Vector([-8.987, -9.838, 5.031])
#     u = Vector([-4.268, -1.861, -8.866])
#     print(v.area_of_parallelogram(u))
#     v = Vector([1.5, 9.547, 3.691])
#     u = Vector([-6.007, 0.124, 5.772])
#     print(v.area_of_triangle(u))

if __name__ == "__main__":
    main()