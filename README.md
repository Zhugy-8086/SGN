# SGN — Structured Gradient Network

SGN（Structured Gradient Network）是一个以整数 / 量化路径为特色的神经网络框架，
提供独特的前向整数编码（HC / MSint）与多种量化反向传播策略（STE / GEF / SR），
用于探索低精度、整数域的深度学习。SGN 完全独立运行，不依赖 PyTorch 等框架。

## 当前仓库内容

本仓库当前**仅公开 SIMD 原语层**（`simd/`）：平台无关接口 + x86/ARM 后端的
高性能点积 / 解码 / 归约 / 字节操作原语，含运行时 CPU 指令集自动检测与回退。

- AVX-512 / AVX-512VNNI / AVX2 / AVX-VNNI / SSSE3 多档内核，多累加器展开
- 运行时 CPUID 检测自动选后端，同一二进制跨 CPU 安全回退（无指令集降级不崩溃）
- 整型原语 bit-exact（与标量逐位一致）；浮点归约标注精度档位供跨后端复现
- 支持 AMD EPYC（AVX-512 服务器加速）、Intel 消费级（AVX2/AVX-VNNI）等平台
- 远程 EPYC 实测：正确性 238/238 全过；dot8/dot4 AVX2 中间档补齐 AMD 适配缺口

## 为什么当前仅公开 SIMD 层

其余功能模块（自动微分引擎、量化训练、Level 调度、HC/MSint 编码等）尚处于
**功能/工程完善阶段**，暂未达到可对外正常使用的完成度，故未随本仓库一同公开。
待各模块达到稳定可用水平后，将分批合并公开。
