import numpy as np

def angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return angle

a = np.array([6,0])
b = np.array([0,0])
c = np.array([0,6])

print(angle(a,b,c))

while(1):
    a1_x = int(input('x von A: '))
    a1_y = int(input('y von A: '))
    b1_x = int(input('x von B: '))
    b1_y = int(input('y von B: '))
    c1_x = int(input('x von C: '))
    c1_y = int(input('y von C: '))
    a2_x = int(input('x von A\': '))
    a2_y = int(input('y von A\': '))
    b2_x = int(input('x von B\': '))
    b2_y = int(input('y von B\': '))

    a1 = np.array([a1_x,a1_y])
    b1 = np.array([b1_x,b1_y])
    c1 = np.array([c1_x,c1_y])
    a2 = np.array([a2_x,a2_y])
    b2 = np.array([b2_x,b2_y])

    delta = a2 - a1
    print('Verschiebung; x: ' + str(delta[0]) + ', y: ' + str(delta[1]))

    b3 = b1 + delta

    rotation = angle(b3, a2, b2)
    print("Drehung: " + str(np.degrees(rotation)))

    c3 = c1 + delta

    a2c3 = c3 - a2

    a2c2 = a2c3 / np.cos(rotation)

    c2 = a2c2 + a2
    print("Punkt C'; x: " + str(c2[0]) + ", y: " + str(c2[1]))

