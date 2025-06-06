// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

// Generated by Fuzzlyn v2.5 on 2025-03-13 04:54:06
// Run on X64 Linux
// Seed: 15128240988293741626-vectort,vector128,vector256,x86aes,x86avx,x86avx2,x86bmi1,x86bmi1x64,x86bmi2,x86bmi2x64,x86fma,x86lzcnt,x86lzcntx64,x86pclmulqdq,x86popcnt,x86popcntx64,x86sse,x86ssex64,x86sse2,x86sse2x64,x86sse3,x86sse41,x86sse41x64,x86sse42,x86sse42x64,x86ssse3,x86x86base
// Reduced from 188.7 KiB to 0.8 KiB in 00:01:09
// Debug: Outputs 0
// Release: Outputs 1
using System;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.CompilerServices;
using Xunit;

public class Runtime_113488
{
    [Fact]
    public static int TestEntryPoint()
    {
        S0 vr0 = default(S0);
        return M4(vr0, Vector256.Create<ushort>(vr0.M7()));
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static int M4(S0 arg0, Vector256<ushort> arg1)
    {
        return arg0.F0 == 0 ? 100 : 101;
    }
    
    private struct S0
    {
        public uint F0;
        [MethodImpl(MethodImplOptions.NoInlining)]
        public ushort M7()
        {
            this.F0 = 1;
            return 0;
        }
    }
}
