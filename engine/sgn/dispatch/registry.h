// registry.h - 内核注册表（一次性能力解析）
//
// 设计背景：内部调研文档（conv2d backward im2col/col2im/transpose 优化调研）
//   节奏 0：搭接口骨架，不改任何内核逻辑。
//
// 能力解析在进程生命周期只执行一次（C++11 magic static，线程安全），
// 执行路径上是零分支的间接调用——回退通道存在但不影响主内核速度。

#pragma once

#include "dispatch/kernel_api.h"

namespace sgn_autograd {

// 返回当前进程活跃的 matmul 内核集合（只读，进程生命周期内不变）。
const KernelSet& kernel_registry();

// 返回当前进程活跃的 conv2d 内核集合（只读，进程生命周期内不变）。
// 节奏 1：仅 dw 端口已实现，dx/db/dycol/col2im/im2col 为 stub（调用即抛）。
const Conv2dKernelSet& conv2d_registry();

}  // namespace sgn_autograd
