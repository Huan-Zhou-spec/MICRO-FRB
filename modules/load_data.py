#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 19:08:46 2026

@author: ubuntu
"""

import os
import time
from datetime import timedelta

# 尝试使用urllib库，这是Python内置的，不需要额外安装
try:
    from urllib.request import urlopen, Request
    from urllib.error import URLError, HTTPError
    USE_URLLIB = True
except ImportError:
    USE_URLLIB = False

def download_with_urllib(url, save_folder="canfar_data", chunk_size=65536):
    """使用Python内置的urllib库下载文件（优化版）"""
    
    # 创建保存文件夹
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # 从URL提取文件名
    filename = url.split('/')[-1]
    filepath = os.path.join(save_folder, filename)
    
    print(f"使用urllib下载 {filename}...")
    print(f"URL: {url}")
    print(f"保存到: {filepath}")
    print(f"块大小: {chunk_size/1024:.0f}KB")
    
    # 开始计时
    start_time = time.time()
    
    try:
        # 创建请求对象，设置用户代理
        req = Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Encoding': 'gzip, deflate, br',  # 支持压缩
            'Accept': '*/*',
            'Connection': 'keep-alive'
        })
        
        # 打开URL连接
        with urlopen(req, timeout=60) as response:
            # 获取文件大小
            file_size = int(response.headers.get('Content-Length', 0))
            
            # 检查是否支持部分内容（断点续传）
            accept_ranges = response.headers.get('Accept-Ranges', 'none')
            print(f"服务器支持断点续传: {accept_ranges != 'none'}")
            
            # 获取内容编码（是否压缩）
            content_encoding = response.headers.get('Content-Encoding', 'none')
            print(f"内容编码: {content_encoding}")
            
            # 计算预计下载时间（假设平均速度100KB/s）
            estimated_time = file_size / (100 * 1024) if file_size > 0 else 0
            
            if file_size > 0:
                print(f"文件大小: {file_size:,} bytes ({file_size/(1024*1024):.2f} MB)")
                if estimated_time > 0:
                    print(f"预计下载时间: {estimated_time:.1f} 秒")
            
            # 读取数据并写入文件
            downloaded = 0
            chunk_start_time = time.time()
            
            with open(filepath, 'wb') as f:
                while True:
                    # 读取指定大小的数据块
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    
                    # 写入文件
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # 计算下载速度
                    chunk_time = time.time() - chunk_start_time
                    if chunk_time > 0.1:  # 每0.1秒更新一次速度
                        speed = len(chunk) / chunk_time / 1024  # KB/s
                        chunk_start_time = time.time()
                        
                        # 显示进度
                        if file_size > 0:
                            progress = (downloaded / file_size) * 100
                            remaining = (file_size - downloaded) / (speed * 1024) if speed > 0 else 0
                            print(f"进度: {progress:.1f}% | 速度: {speed:.1f} KB/s | 剩余: {remaining:.1f}s", end='\r')
                        else:
                            print(f"已下载: {downloaded:,} bytes | 速度: {speed:.1f} KB/s", end='\r')
            
            print("\n" + "="*60)
            
            # 计算总下载时间
            total_time = time.time() - start_time
            total_time_str = str(timedelta(seconds=int(total_time)))
            
            # 计算平均下载速度
            avg_speed = downloaded / total_time / 1024 if total_time > 0 else 0
            
            print(f"✓ 下载完成！")
            print(f"总时间: {total_time_str}")
            print(f"平均速度: {avg_speed:.1f} KB/s")
            
            # 显示文件大小
            actual_size = os.path.getsize(filepath)
            print(f"文件大小: {actual_size / (1024*1024):.2f} MB")
            
            return True
            
    except HTTPError as e:
        print(f"\nHTTP错误: {e.code} - {e.reason}")
    except URLError as e:
        print(f"\nURL错误: {e.reason}")
    except Exception as e:
        print(f"\n错误: {e}")
    
    return False

def download_with_wget(url, save_folder="canfar_data"):
    """使用系统wget命令下载文件（优化版）"""
    
    # 创建保存文件夹
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # 从URL提取文件名
    filename = url.split('/')[-1]
    filepath = os.path.join(save_folder, filename)
    
    print(f"使用wget下载 {filename}...")
    
    # 开始计时
    start_time = time.time()
    
    try:
        # 构建优化的wget命令
        # 参数说明:
        # -c: 断点续传
        # --tries=5: 尝试5次
        # --timeout=60: 超时60秒
        # --waitretry=5: 重试等待5秒
        # --retry-connrefused: 连接被拒绝时也重试
        # --progress=dot: 显示下载进度
        # --limit-rate=0: 不限速
        # -O: 指定输出文件名
        cmd = f'wget -c "{url}" -O "{filepath}" --tries=5 --timeout=60 --waitretry=5 --retry-connrefused --progress=dot --limit-rate=0'
        
        print(f"执行命令: {cmd}")
        print("=" * 60)
        
        # 执行命令
        result = os.system(cmd)
        
        # 计算总下载时间
        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        
        if result == 0:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath) / (1024*1024)
                avg_speed = size * 1024 / total_time if total_time > 0 else 0
                
                print(f"\n" + "="*60)
                print(f"✓ 下载完成！")
                print(f"总时间: {total_time_str}")
                print(f"平均速度: {avg_speed:.1f} KB/s")
                print(f"文件大小: {size:.2f} MB")
                print(f"保存到: {filepath}")
                return True
        else:
            print(f"\n✗ 下载失败，返回码: {result}")
            
    except Exception as e:
        print(f"错误: {e}")
    
    return False

def check_system_command(command):
    """检查系统命令是否可用"""
    try:
        # 使用which命令查找可执行文件
        result = os.system(f"which {command} > /dev/null 2>&1")
        return result == 0
    except:
        return False


def download_canfar_file(
    url=None, 
    save_folder="canfar_downloads",
    benchmark=False,
    method=None,
    auto_select=True
):
    """
    主函数：下载CANFAR文件
    
    参数:
        url (str): 要下载的文件URL，如果为None则使用默认URL
        save_folder (str): 保存文件的文件夹路径
        benchmark (bool): 是否运行基准测试
        method (str): 指定下载方法: 'urllib', 'wget', 'auto' (默认)
        auto_select (bool): 是否自动选择最优下载方法
    
    返回:
        bool: 下载是否成功
    """
    
    # 默认URL
    if url is None:
        url = "https://www.canfar.net/storage/vault/file/AstroDataCitationDOI/CISTI.CANFAR/25.0066/data/dynamic_spectra/data/FRB20180725A_stokesi_dynamic_spectrum.h5"
    
    print("=" * 70)
    print("CANFAR 文件下载器")
    print("=" * 70)
    print(f"文件URL: {url}")
    print(f"保存到: {save_folder}")
    print("=" * 70)
    
    # 创建保存文件夹
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # 开始总计时
    total_start_time = time.time()
    
    # 如果指定了方法，使用指定方法
    if method is not None:
        print(f"\n[指定方法] 使用 {method} 下载...")
        
        if method.lower() == 'urllib' and USE_URLLIB:
            success = download_with_urllib(url, save_folder, 65536)
        elif method.lower() == 'wget':
            if check_system_command("wget"):
                success = download_with_wget(url, save_folder)
            else:
                print("wget不可用")
                success = False
        else:
            print(f"未知或不可用的方法: {method}")
            success = False
            
        # 计算总时间
        total_time = time.time() - total_start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        
        print(f"\n总下载时间: {total_time_str}")
        
        return success
    
    # 询问用户是否要运行基准测试
    if benchmark:
        # 这里可以调用benchmark_download函数（如果已定义）
        print("基准测试功能需要完整的benchmark_download函数")
        # 暂时跳过基准测试，直接使用最优方法
        benchmark = False
    
    if benchmark:
        print("\n运行基准测试...")
        # 这里可以调用benchmark_download函数
        # benchmark_results = benchmark_download(url, save_folder)
        
        # 询问使用哪个方法
        print("\n" + "=" * 70)
        print("请选择下载方法:")
        print("1. urllib (平衡)")
        print("2. wget (稳定)")
        print("3. 退出")
        
        choice = input("请输入选择 (1-3): ").strip()
        
        if choice == '1':
            print("\n[选择] 使用urllib下载 (64KB块)...")
            success = download_with_urllib(url, save_folder, 65536)
        elif choice == '2':
            print("\n[选择] 使用wget下载...")
            success = download_with_wget(url, save_folder)
        else:
            print("退出")
            return False
    else:
        if auto_select:
            # 自动选择最优方法
            print("\n自动选择最优下载方法...")
            
            # 检查aria2是否可用（最快）
            if check_system_command("aria2c"):
                print("检测到aria2，使用多线程下载...")
                # 这里可以调用download_with_aria2函数（如果已定义）
                # 暂时使用wget作为替代
                success = download_with_wget(url, save_folder)
            elif check_system_command("wget"):
                print("检测到wget，使用wget下载...")
                success = download_with_wget(url, save_folder)
            elif USE_URLLIB:
                print("使用urllib下载 (64KB块)...")
                success = download_with_urllib(url, save_folder, 65536)
            else:
                print("没有可用的下载方法!")
                return False
        else:
            # 不使用自动选择，直接尝试urllib
            if USE_URLLIB:
                print("\n使用urllib下载...")
                success = download_with_urllib(url, save_folder, 65536)
            elif check_system_command("wget"):
                print("\n使用wget下载...")
                success = download_with_wget(url, save_folder)
            else:
                print("没有可用的下载方法!")
                return False
    
    # 计算总时间
    total_time = time.time() - total_start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    
    print(f"\n总下载时间: {total_time_str}")
    print("\n" + "=" * 70)
    print("下载任务完成!")
    print("=" * 70)
    
    return success







