#!/usr/bin/env python3
"""
求解正负电荷构成的泊松方程
文件：poisson_equation_student.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def solve_poisson_equation(M: int = 100, target: float = 1e-6, max_iterations: int = 10000) -> Tuple[np.ndarray, int, bool]:
    """
    使用松弛迭代法求解二维泊松方程
    
    参数:
        M (int): 每边的网格点数，默认100
        target (float): 收敛精度，默认1e-6
        max_iterations (int): 最大迭代次数，默认10000
    
    返回:
        tuple: (phi, iterations, converged)
            phi (np.ndarray): 电势分布数组，形状为(M+1, M+1)
            iterations (int): 实际迭代次数
            converged (bool): 是否收敛
    """
    # 网格间距
    h = 1.0
    
    # 初始化电势数组
    phi = np.zeros((M+1, M+1), dtype=float)
    
    # 创建电荷密度数组
    rho = np.zeros((M+1, M+1), dtype=float)
    
    # 设置电荷分布
    # 正电荷区域
    rho[60:80, 20:40] = 1.0
    # 负电荷区域
    rho[20:40, 60:80] = -1.0
    
    # 初始化迭代变量
    delta = 1.0
    iterations = 0
    converged = False
    
    # 创建前一步的电势数组副本
    phi_prev = np.copy(phi)
    
    # 主迭代循环
    while delta > target and iterations < max_iterations:
        # 使用有限差分公式更新内部网格点
        phi[1:-1, 1:-1] = 0.25 * (phi[0:-2, 1:-1] + phi[2:, 1:-1] + 
                                 phi[1:-1, :-2] + phi[1:-1, 2:] + 
                                 h*h * rho[1:-1, 1:-1])
        
        # 计算最大变化量
        delta = np.max(np.abs(phi - phi_prev))
        
        # 更新前一步解
        phi_prev = np.copy(phi)
        
        # 增加迭代计数
        iterations += 1
    
    # 检查是否收敛
    converged = (delta <= target)
    
    return phi, iterations, converged

def visualize_solution(phi: np.ndarray, M: int = 100) -> None:
    """
    可视化电势分布
    
    参数:
        phi (np.ndarray): 电势分布数组
        M (int): 网格大小
    
    功能:
        - 使用 plt.imshow() 显示电势分布
        - 添加颜色条和标签
        - 标注电荷位置
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制电势分布
    im = plt.imshow(phi, extent=[0, M, 0, M], origin='lower', cmap='RdBu_r')
    
    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('Electric Potential (V)', rotation=270, labelpad=20)
    
    # 标注电荷位置
    # 正电荷区域
    plt.fill_between([20, 40], [60, 60], [80, 80], color='red', alpha=0.3, label='Positive Charge (+1)')
    # 负电荷区域
    plt.fill_between([60, 80], [20, 20], [40, 40], color='blue', alpha=0.3, label='Negative Charge (-1)')
    
    # 添加标题和标签
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Electric Potential Distribution from Two Square Charges')
    plt.legend()
    
    # 显示图形
    plt.show()

def analyze_solution(phi: np.ndarray, iterations: int, converged: bool) -> None:
    """
    分析解的统计信息
    
    参数:
        phi (np.ndarray): 电势分布数组
        iterations (int): 迭代次数
        converged (bool): 收敛状态
    
    功能:
        打印解的基本统计信息，如最大值、最小值、迭代次数等
    """
    print("\nSolution Analysis:")
    print(f"Iterations: {iterations}")
    print(f"Converged: {converged}")
    print(f"Maximum potential: {np.max(phi):.6f} V")
    print(f"Minimum potential: {np.min(phi):.6f} V")
    
    # 找到极值位置
    max_pos = np.unravel_index(np.argmax(phi), phi.shape)
    min_pos = np.unravel_index(np.argmin(phi), phi.shape)
    print(f"Maximum potential position: ({max_pos[1]}, {max_pos[0]})")
    print(f"Minimum potential position: ({min_pos[1]}, {min_pos[0]})")

if __name__ == "__main__":
    # 测试代码区域
    print("Solving 2D Poisson equation...")
    
    # 设置参数
    M = 100
    target = 1e-6
    max_iter = 10000
    
    # 调用求解函数
    phi, iterations, converged = solve_poisson_equation(M, target, max_iter)
    
    # 分析结果
    analyze_solution(phi, iterations, converged)
    
    # 可视化结果
    visualize_solution(phi, M)
