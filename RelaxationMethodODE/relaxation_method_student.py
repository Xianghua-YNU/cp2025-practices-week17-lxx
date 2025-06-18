import numpy as np
import matplotlib.pyplot as plt

def solve_ode(h, g, max_iter=10000, tol=1e-6):
    """
    实现松弛迭代法求解常微分方程 d²x/dt² = -g
    边界条件：x(0) = x(10) = 0（抛体运动问题）
    
    参数:
        h (float): 时间步长
        g (float): 重力加速度
        max_iter (int): 最大迭代次数
        tol (float): 收敛容差
    
    返回:
        tuple: (时间数组, 解数组)
    """
    # 初始化时间数组
    t = np.arange(0, 10 + h, h)
    
    # 初始化解数组，边界条件已满足：x[0] = x[-1] = 0
    x = np.zeros(t.size)
    
    # 松弛迭代算法实现
    delta = 1.0  # 初始变化量
    iteration = 0
    
    while delta > tol and iteration < max_iter:
        x_new = np.copy(x)
        # 应用松弛迭代公式更新内部点
        x_new[1:-1] = 0.5 * (h*h*g + x[2:] + x[:-2])
        
        # 计算最大变化量
        delta = np.max(np.abs(x_new - x))
        
        # 更新解
        x = x_new
        iteration += 1
    
    return t, x

if __name__ == "__main__":
    # 测试参数
    h = 10 / 100  # 时间步长
    g = 9.8       # 重力加速度
    
    # 调用求解函数
    t, x = solve_ode(h, g)
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(t, x, 'b-', linewidth=2)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Height (m)', fontsize=12)
    plt.title('Projectile Motion Trajectory (Relaxation Method)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 标记最大高度和时间
    max_height = np.max(x)
    max_time = t[np.argmax(x)]
    plt.annotate(f'Max height: {max_height:.2f}m at {max_time:.2f}s',
                 xy=(max_time, max_height), xytext=(max_time+1, max_height-10),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10)
    
    plt.show()
