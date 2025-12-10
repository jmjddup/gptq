import argparse
import subprocess
import sys

def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description='Run OPT quantization with different modes')
    parser.add_argument('-m', 
                        type=int, 
                        choices=[0, 1, 2, 3, 4], 
                        required=True, 
                        help='Mode selection: 0 for default run, 1 for 4-bit quantization')
    parser.add_argument('--device', 
                        type=str, 
                        default='0', 
                        help='CUDA device ID (default: 0)')
    parser.add_argument('--model', 
                        type=str, 
                        default='/root/fshare/models/facebook/opt-125m', 
                        help='Model name (default: /root/fshare/models/facebook/opt-125m)')
    parser.add_argument('--dataset', 
                        type=str, 
                        default='wikitext2', 
                        help='Dataset name (default: wikitext2)')
    
    args = parser.parse_args()

    # 2. 构建基础命令
    base_cmd = [
        'python', 
        'opt.py', 
        args.model, 
        args.dataset
    ]

    # 3. 根据模式添加量化参数
    if args.m == 1:
        base_cmd.extend(['--wbits', '4', '--nearest'])
    elif args.m == 2:
        base_cmd.extend(['--wbits', '4'])
    elif args.m == 3:
        base_cmd.extend(['--wbits', '4', '--groupsize 1024'])
    elif args.m == 4:
        base_cmd.extend(['--wbits', '4', '--act-order'])

    # 4. 构建包含 CUDA 设备的完整命令（兼容 Windows/Linux）
    full_cmd = f'CUDA_VISIBLE_DEVICES={args.device} ' + ' '.join(base_cmd)
    shell = True

    # 5. 打印并执行命令
    print(f'Executing command: {full_cmd}')
    try:
        result = subprocess.run(full_cmd, 
                                shell=shell, 
                                check=True, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                text=True)
        # 输出执行结果
        print('\n=== Command Output ===')
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f'\n=== Error Executing Command ===')
        print(f'Return code: {e.returncode}')
        print(f'Stderr: {e.stderr}')
        sys.exit(1)

if __name__ == '__main__':
    main()