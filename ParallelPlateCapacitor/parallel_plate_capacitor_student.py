"""平行板电容器电势分布模拟"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def solve_laplace_jacobi(xgrid, ygrid, w, d, tol=1e-5):
    """
    使用Jacobi迭代法求解拉普拉斯方程
    
    参数:
        xgrid (int): x方向网格点数
        ygrid (int): y方向网格点数
        w (int): 平行板宽度
        d (int): 平行板间距
        tol (float): 收敛容差
    
    返回:
        tuple: (potential_array, iterations, convergence_history)
    """
    # 初始化电势网格
    u = np.zeros((ygrid, xgrid))
    
    # 计算平行板位置
    xL = (xgrid - w) // 2
    xR = (xgrid + w) // 2
    yB = (ygrid - d) // 2
    yT = (ygrid + d) // 2
    
    # 设置边界条件
    u[yT, xL:xR+1] = 100.0    # 上板 +100V
    u[yB, xL:xR+1] = -100.0   # 下板 -100V
    
    # 边界条件：四周接地
    u[0, :] = 0    # 下边界
    u[-1, :] = 0   # 上边界
    u[:, 0] = 0    # 左边界
    u[:, -1] = 0   # 右边界
    
    iterations = 0
    max_diff = 1.0
    convergence_history = []
    
    # Jacobi迭代
    while max_diff > tol:
        u_old = u.copy()
        max_diff = 0.0
        
        # 更新内部点
        for i in range(1, ygrid-1):
            for j in range(1, xgrid-1):
                # 跳过平行板区域
                if (i == yT and xL <= j <= xR) or (i == yB and xL <= j <= xR):
                    continue
                
                # Jacobi迭代公式
                u[i, j] = 0.25 * (u_old[i+1, j] + u_old[i-1, j] + 
                                 u_old[i, j+1] + u_old[i, j-1])
                
                # 计算最大变化量
                diff = abs(u[i, j] - u_old[i, j])
                if diff > max_diff:
                    max_diff = diff
        
        iterations += 1
        convergence_history.append(max_diff)
        
        # 防止无限循环
        if iterations > 10000:
            break
    
    return u, iterations, convergence_history

def solve_laplace_sor(xgrid, ygrid, w, d, omega=1.25, Niter=1000, tol=1e-5):
    """
    实现SOR算法求解平行板电容器的电势分布
    
    参数:
        xgrid (int): x方向网格点数
        ygrid (int): y方向网格点数
        w (int): 平行板宽度
        d (int): 平行板间距
        omega (float): 松弛因子
        Niter (int): 最大迭代次数
        tol (float): 收敛容差
    返回:
        tuple: (电势分布数组, 迭代次数, 收敛历史)
    """
    # 初始化电势网格
    u = np.zeros((ygrid, xgrid))
    
    # 计算平行板位置
    xL = (xgrid - w) // 2
    xR = (xgrid + w) // 2
    yB = (ygrid - d) // 2
    yT = (ygrid + d) // 2
    
    # 设置边界条件
    u[yT, xL:xR+1] = 100.0    # 上板 +100V
    u[yB, xL:xR+1] = -100.0   # 下板 -100V
    
    # 边界条件：四周接地
    u[0, :] = 0    # 下边界
    u[-1, :] = 0   # 上边界
    u[:, 0] = 0    # 左边界
    u[:, -1] = 0   # 右边界
    
    iterations = 0
    max_diff = 1.0
    convergence_history = []
    
    # SOR迭代
    for iterations in range(Niter):
        max_diff = 0.0
        
        # 更新内部点
        for i in range(1, ygrid-1):
            for j in range(1, xgrid-1):
                # 跳过平行板区域
                if (i == yT and xL <= j <= xR) or (i == yB and xL <= j <= xR):
                    continue
                
                # 计算残差
                residual = 0.25 * (u[i+1, j] + u[i-1, j] + 
                                  u[i, j+1] + u[i, j-1]) - u[i, j]
                
                # SOR更新公式
                new_val = u[i, j] + omega * residual
                
                # 计算最大变化量
                diff = abs(new_val - u[i, j])
                if diff > max_diff:
                    max_diff = diff
                
                u[i, j] = new_val
        
        convergence_history.append(max_diff)
        
        # 检查收敛
        if max_diff < tol:
            break
    
    return u, iterations+1, convergence_history

def plot_results(x, y, u, method_name):
    """
    绘制三维电势分布、等势线和电场线
    
    参数:
        x (array): X坐标数组
        y (array): Y坐标数组
        u (array): 电势分布数组
        method_name (str): 方法名称
    """
    # 创建图形
    fig = plt.figure(figsize=(12, 6))
    
    # 1. 三维电势分布图
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))
    ax1.plot_wireframe(X, Y, u, rstride=2, cstride=2, linewidth=0.5)
    ax1.set_title(f'3D Potential Distribution ({method_name})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Potential (V)')
    
    # 2. 等势线和电场线图
    ax2 = fig.add_subplot(122)
    
    # 计算电场
    Ey, Ex = np.gradient(-u)
    
    # 绘制等势线
    levels = np.linspace(-100, 100, 21)
    contour = ax2.contour(X, Y, u, levels=levels, cmap='jet')
    plt.colorbar(contour, ax=ax2, label='Potential (V)')
    
    # 绘制电场线
    ax2.streamplot(X, Y, Ex, Ey, color='k', linewidth=0.5, density=1.5)
    
    ax2.set_title(f'Equipotential Lines & Electric Field ({method_name})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 参数设置
    xgrid, ygrid = 50, 50  # 网格大小
    w, d = 10, 10          # 平行板宽度和间距
    
    # 使用Jacobi方法求解
    start_time = time.time()
    u_jacobi, iter_jacobi, conv_jacobi = solve_laplace_jacobi(xgrid, ygrid, w, d)
    jacobi_time = time.time() - start_time
    
    # 使用SOR方法求解
    start_time = time.time()
    u_sor, iter_sor, conv_sor = solve_laplace_sor(xgrid, ygrid, w, d, omega=1.5)
    sor_time = time.time() - start_time
    
    # 打印结果比较
    print(f"Jacobi Method: {iter_jacobi} iterations, {jacobi_time:.3f} seconds")
    print(f"SOR Method: {iter_sor} iterations, {sor_time:.3f} seconds")
    
    # 绘制收敛历史
    plt.figure()
    plt.semilogy(conv_jacobi, label='Jacobi')
    plt.semilogy(conv_sor, label='SOR (ω=1.5)')
    plt.xlabel('Iteration')
    plt.ylabel('Maximum Change')
    plt.title('Convergence History')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 绘制结果
    x = np.arange(xgrid)
    y = np.arange(ygrid)
    plot_results(x, y, u_jacobi, 'Jacobi Method')
    plot_results(x, y, u_sor, 'SOR Method')
