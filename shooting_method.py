import numpy as np
import matplotlib.pyplot as plt

t0 = 0
tf = 10
mv = 757
s0 = [0, 0, 0, 0]
target = [500, 0]
K = 0.005
gravity = -9.81
#launch angle from horrizontal
phi = 0.05
def f_prime(t, s):
    speed = np.hypot(s[1], s[3])
    a = -np.sign(s[1])*K*speed
    b = -np.sign(s[3])*K*speed + gravity/s[3]
    A = np.array(
            [
                [0,1,0,0],
                [0,a,0,0],
                [0,0,0,1],
                [0,0,0,b] 
            ]
            , dtype = np.float64)
    res = np.dot(A, s)
    return res

def impact_event(t, s):
    return s[2]
impact_event.terminal = True
impact_event.direction = -1

def objective(phi, plot=False):
    from scipy.integrate import solve_ivp
    assert(1 == len(phi))
    s0[1] = mv * np.cos(*phi)
    s0[3] = mv * np.sin(*phi)
    sol = solve_ivp(f_prime, [t0, tf], \
            s0, events = impact_event, dense_output = True)
    t_event = sol.t_events[0]
    t_eval = np.linspace(t0, *t_event, 1000)
    res = sol.sol(t_eval)
    r =  res[0][-1] - target[0], res[2][-1] - target[1]
    if(True==plot):
        print("start: ", s0)
        print("error: ", r)
        plt.plot(res[0], res[2])
        plt.show()
    return np.abs(r[0]) + np.abs(r[1])

if __name__ == "__main__":
    from scipy.optimize import fsolve
    phi = fsolve(objective, phi)
    objective(phi, plot=True)
