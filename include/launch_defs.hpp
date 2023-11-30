#pragma once

#ifndef _BLOCK_M
#define _BLOCK_M 32
#endif

#ifndef _BLOCK_N
#define _BLOCK_N 32
#endif

#ifndef _BLOCK_K
#define _BLOCK_K 32
#endif

#ifndef _Warps
#define _Warps 1
#endif

#ifndef LAUNCH_NAME
#define LAUNCH_NAME _kernel
#endif

#ifndef _VecLoad
#define _VecLoad 1
#endif

#ifndef _InnerK
#define _InnerK 16
#endif