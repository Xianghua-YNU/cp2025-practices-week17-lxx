"""
波动方程FTCS解
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def u_t(x, C=1, d=0.1, sigma=0.3, L=1):
    """
    计算初始速度剖面 psi(x)。

    参数:
        x (np.ndarray): 位置数组。
        C (float): 振幅常数。
        d (float): 指数项的偏移量。
        sigma (float): 指数项的宽度。
        L (float): 弦的长度。
    返回:
        np.ndarray: 初始速度剖面。
    """
    return C * x * (L - x) / L**2 * np.exp(-(x - d)**2 / (2 * sigma**2))

def solve_wave_equation_ftcs(parameters):
    """
    使用FTCS有限差分法求解一维波动方程。
    
    参数:
        parameters (dict): 包含以下参数的字典：
            - 'a': 波速 (m/s)。
            - 'L': 弦的长度 (m)。
            - 'd': 初始速度剖面的偏移量 (m)。
            - 'C': 初始速度剖面的振幅常数 (m/s)。
            - 'sigma': 初始速度剖面的宽度 (m)。
            - 'dx': 空间步长 (m)。
            - 'dt': 时间步长 (s)。
            - 'total_time': 总模拟时间 (s)。
    返回:
        tuple: 包含以下内容的元组：
            - np.ndarray: 解数组 u(x, t)。
            - np.ndarray: 空间数组 x。
            - np.ndarray: 时间数组 t。
    """
    # 从参数字典中获取参数
    a = parameters['a']
    L = parameters['L']
    dx = parameters['dx']
    dt = parameters['dt']
    total_time = parameters['total_time']
    
    # 计算空间和时间网格
    x = np.arange(0, L + dx, dx)
    t = np.arange(0, total_time + dt, dt)
    
    # 初始化解数组
    u = np.zeros((len(x), len(t)))
    
    # 计算稳定性条件
    c = (a * dt / dx)**2
    if c >= 1:
        print(f"Warning: Stability condition violated (c = {c:.4f} >= 1)")
    
    # 应用初始条件 u(x,0) = 0 (已经由np.zeros初始化)
    
    # 计算第一个时间步 u(x,1)
    u[:, 1] = u_t(x, C=parameters['C'], d=parameters['d'], 
                 sigma=parameters['sigma'], L=parameters['L']) * dt
    
    # FTCS主算法
    for j in range(1, len(t)-1):
        for i in range(1, len(x)-1):
            u[i, j+1] = c * (u[i+1, j] + u[i-1, j]) + 2*(1 - c)*u[i, j] - u[i, j-1]
        
        # 应用边界条件
        u[0, j+1] = 0
        u[-1, j+1] = 0
    
    return u, x, t

if __name__ == "__main__":
    # 演示和测试
    params = {
        'a': 100,
        'L': 1,
        'd': 0.1,
        'C': 1,
        'sigma': 0.3,
        'dx': 0.01,
        'dt': 5e-5,
        'total_time': 0.1
    }
    u_sol, x_sol, t_sol = solve_wave_equation_ftcs(params)

    # 创建动画
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, xlim=(0, params['L']), ylim=(u_sol.min() * 1.1, u_sol.max() * 1.1))
    line, = ax.plot([], [], 'g-', lw=2)
    ax.set_title("1D Wave Equation (FTCS)")
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Displacement")

    def update(frame):
        line.set_data(x_sol, u_sol[:, frame])
        return line,

    # 创建动画，每帧间隔50毫秒
    ani = FuncAnimation(fig, update, frames=len(t_sol), interval=50, blit=True)
    
    # 保存动画为GIF（可选）
    # ani.save('wave_animation.gif', writer='pillow', fps=30)
    
    plt.show()
