#!/usr/bin/env python3
"""
修正后的泊松方程求解器
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def solve_poisson_equation(M: int = 100, target: float = 1e-6, max_iterations: int = 10000) -> Tuple[np.ndarray, int, bool]:
    """
    修正后的松弛迭代法求解二维泊松方程
    
    主要修正：
    1. 电荷区域坐标根据M值动态计算
    2. 严格保证边界条件
    3. 优化迭代过程
    """
    h = 1.0  # 网格间距
    
    # 初始化电势数组
    phi = np.zeros((M+1, M+1), dtype=float)
    
    # 创建电荷密度数组
    rho = np.zeros((M+1, M+1), dtype=float)
    
    # 动态计算电荷区域(原区域按比例缩放)
    pos_start_x, pos_end_x = int(0.6*M), int(0.8*M)
    pos_start_y, pos_end_y = int(0.2*M), int(0.4*M)
    neg_start_x, neg_end_x = int(0.2*M), int(0.4*M)
    neg_start_y, neg_end_y = int(0.6*M), int(0.8*M)
    
    # 设置电荷分布
    rho[pos_start_x:pos_end_x, pos_start_y:pos_end_y] = 1.0   # 正电荷
    rho[neg_start_x:neg_end_x, neg_start_y:neg_end_y] = -1.0  # 负电荷
    
    # 迭代变量
    delta = 1.0
    iterations = 0
    converged = False
    
    # 主迭代循环
    while delta > target and iterations < max_iterations:
        phi_prev = phi.copy()
        
        # 更新内部点
        phi[1:-1, 1:-1] = 0.25 * (phi_prev[2:, 1:-1] + phi_prev[:-2, 1:-1] + 
                                 phi_prev[1:-1, 2:] + phi_prev[1:-1, :-2] + 
                                 h*h * rho[1:-1, 1:-1])
        
        # 严格保证边界条件
        phi[0, :] = 0.0    # 上边界
        phi[-1, :] = 0.0   # 下边界
        phi[:, 0] = 0.0    # 左边界
        phi[:, -1] = 0.0   # 右边界
        
        delta = np.max(np.abs(phi - phi_prev))
        iterations += 1
    
    converged = delta <= target
    return phi, iterations, converged

def visualize_solution(phi: np.ndarray, M: int = 100) -> None:
    """可视化函数保持不变"""
    plt.figure(figsize=(10, 8))
    im = plt.imshow(phi.T, extent=[0, M, 0, M], origin='lower', cmap='RdBu_r')
    plt.colorbar(im).set_label('Electric Potential (V)')
    
    # 动态计算电荷区域
    pos_coords = (int(0.2*M), int(0.4*M), int(0.6*M), int(0.8*M))
    neg_coords = (int(0.6*M), int(0.8*M), int(0.2*M), int(0.4*M))
    
    plt.fill_betweenx([neg_coords[2], neg_coords[3]], neg_coords[0], neg_coords[1], 
                     color='blue', alpha=0.3, label='Negative Charge')
    plt.fill_betweenx([pos_coords[2], pos_coords[3]], pos_coords[0], pos_coords[1], 
                     color='red', alpha=0.3, label='Positive Charge')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Electric Potential Distribution')
    plt.legend()
    plt.show()

def analyze_solution(phi: np.ndarray, iterations: int, converged: bool) -> None:
    """分析函数保持不变"""
    print(f"\nSolution Analysis (Grid {phi.shape[0]-1}x{phi.shape[1]-1}):")
    print(f"Iterations: {iterations}, Converged: {converged}")
    print(f"Potential Range: [{np.min(phi):.4f}, {np.max(phi):.4f}] V")
    
    max_pos = np.unravel_index(np.argmax(phi), phi.shape)
    min_pos = np.unravel_index(np.argmin(phi), phi.shape)
    print(f"Max at ({max_pos[1]}, {max_pos[0]}), Min at ({min_pos[1]}, {min_pos[0]})")

if __name__ == "__main__":
    # 测试不同网格尺寸
    for M in [20, 50, 100]:
        print(f"\nSolving for M = {M}")
        phi, iterations, converged = solve_poisson_equation(M)
        analyze_solution(phi, iterations, converged)
        if M == 100:  # 只可视化最大的网格
            visualize_solution(phi, M)
