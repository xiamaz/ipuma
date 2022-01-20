	.text
	.allow_optimizations
	.file	"3"
	.section	.text.__runCodelet_SWAffine,"ax",@progbits
	.globl	__runCodelet_SWAffine           # -- Begin function __runCodelet_SWAffine
	.p2align	2
	.type	__runCodelet_SWAffine,@function
__runCodelet_SWAffine:                  # @__runCodelet_SWAffine
.Lfunc_begin0:
# %bb.0:
	mov	$m11, $m12
	mov	$m0, $m13
	call $m10, _ZN8SWAffine7computeEv
	exitnz $m0
.Lfunc_end0:
	.size	__runCodelet_SWAffine, .Lfunc_end0-__runCodelet_SWAffine
	.section	.stack_sizes,"o",@progbits,.text.__runCodelet_SWAffine
	.long	.Lfunc_begin0
	.byte	0
	.section	.text.__runCodelet_SWAffine,"ax",@progbits
                                        # -- End function
	.section	.text._ZN8SWAffine7computeEv,"axG",@progbits,_ZN8SWAffine7computeEv,comdat
	.weak	_ZN8SWAffine7computeEv          # -- Begin function _ZN8SWAffine7computeEv
	.p2align	2
	.type	_ZN8SWAffine7computeEv,@function
_ZN8SWAffine7computeEv:                 # @_ZN8SWAffine7computeEv
.Lfunc_begin1:
# %bb.0:
	add $m11, $m11, -112
	sub $m6, $m12, $m11
	cmpult $m6, $m6, __worker_stack_max_growth_plus_one
	brz $m6, _stackoverflow
	st32 $m8, $m11, $m15, 3                 # 4-byte Folded Spill
	st32 $m9, $m11, $m15, 2                 # 4-byte Folded Spill
	st32 $m10, $m11, $m15, 1                # 4-byte Folded Spill
	st32 $m7, $m11, $m15, 0                 # 4-byte Folded Spill
	mov	$m7, $m0
	ld32 $m0, $m7, $m15, 4
	ld32 $m0, $m0, $m15, 0
	brz $m0, .LBB1_15
# %bb.1:
	st32 $m15, $m11, $m15, 15               # 4-byte Folded Spill
	st32 $m15, $m11, $m15, 9                # 4-byte Folded Spill
	mov	$m2, $m15
	ld32 $m0, $m7, $m15, 5
	ld32 $m1, $m7, $m15, 6
	ld32 $m0, $m0, $m15, 0
	ld32 $m1, $m1, $m15, 0
	st32 $m1, $m11, $m15, 23                # 4-byte Folded Spill
	add $m3, $m1, $m0
	st32 $m7, $m11, $m15, 7                 # 4-byte Folded Spill
	st32 $m3, $m11, $m15, 24                # 4-byte Folded Spill
.LBB1_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_7 Depth 2
                                        #       Child Loop BB1_8 Depth 3
                                        #     Child Loop BB1_11 Depth 2
                                        #       Child Loop BB1_12 Depth 3
	ld32 $m0, $m7, $m15, 11
	ld32 $m0, $m0, $m15, $m2
	st32 $m0, $m11, $m15, 8                 # 4-byte Folded Spill
	brz $m0, .LBB1_15
# %bb.3:                                #   in Loop: Header=BB1_2 Depth=1
	ld32 $m0, $m7, $m15, 12
	ld32 $m8, $m0, $m15, $m2
	brz $m8, .LBB1_15
# %bb.4:                                #   in Loop: Header=BB1_2 Depth=1
	st32 $m2, $m11, $m15, 6                 # 4-byte Folded Spill
	ld32 $m0, $m7, $m15, 8
	ld32 $m1, $m0, $m15, 0
	ld32 $m0, $m7, $m15, 0
	shl $m2, $m1, 2
	mov	$m1, $m15
	call $m10, memset
	cmpslt $m0, $m8, 1
	st32 $m8, $m11, $m15, 5                 # 4-byte Folded Spill
	brnz $m0, .LBB1_5
# %bb.6:                                #   in Loop: Header=BB1_2 Depth=1
	st32 $m15, $m11, $m15, 22               # 4-byte Folded Spill
	st32 $m15, $m11, $m15, 14               # 4-byte Folded Spill
	st32 $m15, $m11, $m15, 16               # 4-byte Folded Spill
	st32 $m15, $m11, $m15, 17               # 4-byte Folded Spill
	ld32 $m0, $m7, $m15, 5
	st32 $m0, $m11, $m15, 12                # 4-byte Folded Spill
	add $m8, $m8, -1
.LBB1_7:                                #   Parent Loop BB1_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_8 Depth 3
	st32 $m8, $m11, $m15, 13                # 4-byte Folded Spill
	mov	$m3, $m15
	st32 $m15, $m11, $m15, 26               # 4-byte Folded Spill
	mov	$m1, $m15
	ld32 $m5, $m11, $m15, 8                 # 4-byte Folded Reload
	ld32 $m0, $m11, $m15, 12                # 4-byte Folded Reload
	ld32 $m10, $m0, $m15, 0
	ld32 $m6, $m7, $m15, 0
	ld32 $m2, $m7, $m15, 1
	ld32 $m8, $m7, $m15, 9
	ld32 $m9, $m11, $m15, 9                 # 4-byte Folded Reload
	add $m0, $m8, $m9
	st32 $m0, $m11, $m15, 21                # 4-byte Folded Spill
	ld32 $m0, $m7, $m15, 2
	st32 $m0, $m11, $m15, 20                # 4-byte Folded Spill
	ld32 $m0, $m7, $m15, 10
	st32 $m0, $m11, $m15, 19                # 4-byte Folded Spill
	ld32 $m7, $m11, $m15, 15                # 4-byte Folded Reload
	ld32 $m0, $m11, $m15, 22                # 4-byte Folded Reload
	add $m0, $m0, $m7
	st32 $m0, $m11, $m15, 18                # 4-byte Folded Spill
	st32 $m3, $m11, $m15, 25                # 4-byte Folded Spill
	mov	$m8, $m3
	ld32 $m3, $m11, $m15, 16                # 4-byte Folded Reload
	ld32 $m0, $m11, $m15, 14                # 4-byte Folded Reload
	ld32 $m4, $m11, $m15, 17                # 4-byte Folded Reload
	st32 $m5, $m11, $m15, 27                # 4-byte Folded Spill
.LBB1_8:                                #   Parent Loop BB1_2 Depth=1
                                        #     Parent Loop BB1_7 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	ld32 $m7, $m11, $m15, 24                # 4-byte Folded Reload
	add $m1, $m7, $m1
	ld32 $m5, $m11, $m15, 23                # 4-byte Folded Reload
	add $m10, $m10, $m5
	max $m10, $m1, $m10
	ld32 $m1, $m6, $m15, 0
	add $m1, $m7, $m1
	ld32 $m7, $m2, $m15, 0
	add $m7, $m7, $m5
	max $m1, $m1, $m7
	st32step $m1, $m15, $m2+=, 1
	ld32 $m5, $m11, $m15, 21                # 4-byte Folded Reload
	ldz8step $m7, $m5, $m8+=, 1
	ld32 $m5, $m11, $m15, 20                # 4-byte Folded Reload
	ld32 $m7, $m5, $m15, $m7
	ld32 $m5, $m11, $m15, 19                # 4-byte Folded Reload
	ld32 $m9, $m11, $m15, 18                # 4-byte Folded Reload
	ldz8 $m9, $m5, $m15, $m9
	ld32 $m7, $m7, $m15, $m9
	ld32 $m5, $m11, $m15, 26                # 4-byte Folded Reload
	add $m5, $m7, $m5
	max $m5, $m5, $m10
	max $m1, $m5, $m1
	max $m1, $m1, 0
	ld32 $m5, $m6, $m15, 0
	st32 $m5, $m11, $m15, 26                # 4-byte Folded Spill
	st32step $m1, $m15, $m6+=, 1
	ld32 $m5, $m11, $m15, 27                # 4-byte Folded Reload
	add $m5, $m5, -1
	cmpult $m7, $m4, $m1
	movnz	$m4, $m7, $m1
	ld32 $m9, $m11, $m15, 25                # 4-byte Folded Reload
	movnz	$m3, $m7, $m9
	ld32 $m9, $m11, $m15, 22                # 4-byte Folded Reload
	movnz	$m0, $m7, $m9
	st32 $m8, $m11, $m15, 25                # 4-byte Folded Spill
	st32 $m5, $m11, $m15, 27                # 4-byte Folded Spill
	brnz $m5, .LBB1_8
# %bb.9:                                #   in Loop: Header=BB1_7 Depth=2
	st32 $m4, $m11, $m15, 17                # 4-byte Folded Spill
	st32 $m0, $m11, $m15, 14                # 4-byte Folded Spill
	st32 $m3, $m11, $m15, 16                # 4-byte Folded Spill
	ld32 $m8, $m11, $m15, 13                # 4-byte Folded Reload
	ld32 $m0, $m11, $m15, 22                # 4-byte Folded Reload
	add $m0, $m0, 1
	st32 $m0, $m11, $m15, 22                # 4-byte Folded Spill
	ld32 $m7, $m11, $m15, 7                 # 4-byte Folded Reload
	brnzdec $m8, .LBB1_7
	bri .LBB1_10
.LBB1_5:                                #   in Loop: Header=BB1_2 Depth=1
	st32 $m15, $m11, $m15, 17               # 4-byte Folded Spill
	st32 $m15, $m11, $m15, 16               # 4-byte Folded Spill
	st32 $m15, $m11, $m15, 14               # 4-byte Folded Spill
.LBB1_10:                               #   in Loop: Header=BB1_2 Depth=1
	ld32 $m1, $m7, $m15, 13
	ld32 $m2, $m11, $m15, 6                 # 4-byte Folded Reload
	ld32 $m0, $m11, $m15, 17                # 4-byte Folded Reload
	stm32 $m0, $m1, $m2
	ld32 $m0, $m7, $m15, 8
	ld32 $m1, $m0, $m15, 0
	ld32 $m0, $m7, $m15, 0
	shl $m2, $m1, 2
	mov	$m1, $m15
	call $m10, memset
	ld32 $m0, $m7, $m15, 8
	ld32 $m1, $m0, $m15, 0
	ld32 $m0, $m7, $m15, 1
	shl $m2, $m1, 2
	mov	$m1, $m15
	call $m10, memset
	st32 $m15, $m11, $m15, 25               # 4-byte Folded Spill
	st32 $m15, $m11, $m15, 17               # 4-byte Folded Spill
	mov	$m4, $m15
	ld32 $m0, $m11, $m15, 14                # 4-byte Folded Reload
	sort4x16lo $m2, $m0, $m15
	ld32 $m0, $m11, $m15, 16                # 4-byte Folded Reload
	sort4x16lo $m6, $m0, $m15
	add $m8, $m2, 1
	ld32 $m0, $m7, $m15, 5
	st32 $m0, $m11, $m15, 13                # 4-byte Folded Spill
	ld32 $m1, $m7, $m15, 0
	ld32 $m3, $m7, $m15, 1
	ld32 $m5, $m7, $m15, 9
	ld32 $m0, $m11, $m15, 9                 # 4-byte Folded Reload
	add $m0, $m5, $m0
	st32 $m0, $m11, $m15, 20                # 4-byte Folded Spill
	ld32 $m0, $m7, $m15, 2
	st32 $m0, $m11, $m15, 19                # 4-byte Folded Spill
	ld32 $m0, $m7, $m15, 10
	st32 $m0, $m11, $m15, 18                # 4-byte Folded Spill
	add $m8, $m8, -1
	shl $m5, $m6, 2
	add $m0, $m1, $m5
	st32 $m0, $m11, $m15, 12                # 4-byte Folded Spill
	add $m0, $m3, $m5
	st32 $m0, $m11, $m15, 11                # 4-byte Folded Spill
	st32 $m6, $m11, $m15, 14                # 4-byte Folded Spill
	add $m0, $m6, 1
	st32 $m0, $m11, $m15, 10                # 4-byte Folded Spill
	st32 $m2, $m11, $m15, 4                 # 4-byte Folded Spill
	st32 $m2, $m11, $m15, 22                # 4-byte Folded Spill
	ld32 $m3, $m11, $m15, 24                # 4-byte Folded Reload
.LBB1_11:                               #   Parent Loop BB1_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_12 Depth 3
	st32 $m8, $m11, $m15, 16                # 4-byte Folded Spill
	st32 $m15, $m11, $m15, 27               # 4-byte Folded Spill
	mov	$m8, $m15
	ld32 $m1, $m11, $m15, 10                # 4-byte Folded Reload
	ld32 $m0, $m11, $m15, 13                # 4-byte Folded Reload
	ld32 $m7, $m0, $m15, 0
	add $m1, $m1, -1
	ld32 $m0, $m11, $m15, 15                # 4-byte Folded Reload
	ld32 $m2, $m11, $m15, 22                # 4-byte Folded Reload
	add $m0, $m2, $m0
	st32 $m0, $m11, $m15, 21                # 4-byte Folded Spill
	ld32 $m9, $m11, $m15, 11                # 4-byte Folded Reload
	ld32 $m10, $m11, $m15, 12               # 4-byte Folded Reload
	ld32 $m6, $m11, $m15, 14                # 4-byte Folded Reload
	st32 $m6, $m11, $m15, 26                # 4-byte Folded Spill
	ld32 $m2, $m11, $m15, 25                # 4-byte Folded Reload
	ld32 $m5, $m11, $m15, 17                # 4-byte Folded Reload
.LBB1_12:                               #   Parent Loop BB1_2 Depth=1
                                        #     Parent Loop BB1_11 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	add $m8, $m3, $m8
	ld32 $m0, $m11, $m15, 23                # 4-byte Folded Reload
	add $m7, $m7, $m0
	max $m7, $m8, $m7
	st32 $m7, $m11, $m15, 25                # 4-byte Folded Spill
	ld32 $m8, $m10, $m15, 0
	add $m8, $m3, $m8
	ld32 $m3, $m9, $m15, 0
	add $m3, $m3, $m0
	max $m3, $m8, $m3
	st32step $m3, $m15, $m9+=, -1
	ld32 $m0, $m11, $m15, 20                # 4-byte Folded Reload
	ldz8step $m8, $m0, $m6+=, -1
	ld32 $m0, $m11, $m15, 19                # 4-byte Folded Reload
	ld32 $m8, $m0, $m15, $m8
	ld32 $m0, $m11, $m15, 21                # 4-byte Folded Reload
	ld32 $m7, $m11, $m15, 18                # 4-byte Folded Reload
	ldz8 $m0, $m7, $m15, $m0
	ld32 $m7, $m11, $m15, 25                # 4-byte Folded Reload
	ld32 $m0, $m8, $m15, $m0
	ld32 $m8, $m11, $m15, 27                # 4-byte Folded Reload
	add $m0, $m0, $m8
	max $m0, $m0, $m7
	max $m0, $m0, $m3
	max $m8, $m0, 0
	ld32 $m0, $m10, $m15, 0
	st32 $m0, $m11, $m15, 27                # 4-byte Folded Spill
	st32step $m8, $m15, $m10+=, -1
	cmpult $m0, $m4, $m8
	movnz	$m4, $m0, $m8
	ld32 $m3, $m11, $m15, 26                # 4-byte Folded Reload
	movnz	$m5, $m0, $m3
	ld32 $m3, $m11, $m15, 22                # 4-byte Folded Reload
	movnz	$m2, $m0, $m3
	ld32 $m3, $m11, $m15, 24                # 4-byte Folded Reload
	st32 $m6, $m11, $m15, 26                # 4-byte Folded Spill
	brnzdec $m1, .LBB1_12
# %bb.13:                               #   in Loop: Header=BB1_11 Depth=2
	st32 $m5, $m11, $m15, 17                # 4-byte Folded Spill
	st32 $m2, $m11, $m15, 25                # 4-byte Folded Spill
	ld32 $m8, $m11, $m15, 16                # 4-byte Folded Reload
	ld32 $m0, $m11, $m15, 22                # 4-byte Folded Reload
	add $m0, $m0, -1
	st32 $m0, $m11, $m15, 22                # 4-byte Folded Spill
	brnzdec $m8, .LBB1_11
# %bb.14:                               #   in Loop: Header=BB1_2 Depth=1
	ld32 $m0, $m11, $m15, 14                # 4-byte Folded Reload
	shl $m0, $m0, 16
	ld32 $m1, $m11, $m15, 17                # 4-byte Folded Reload
	sort4x16lo $m1, $m1, $m15
	or $m0, $m0, $m1
	ld32 $m7, $m11, $m15, 7                 # 4-byte Folded Reload
	ld32 $m1, $m7, $m15, 15
	ld32 $m2, $m11, $m15, 6                 # 4-byte Folded Reload
	stm32 $m0, $m1, $m2
	ld32 $m0, $m11, $m15, 4                 # 4-byte Folded Reload
	shl $m0, $m0, 16
	ld32 $m1, $m11, $m15, 25                # 4-byte Folded Reload
	sort4x16lo $m1, $m1, $m15
	or $m0, $m0, $m1
	ld32 $m1, $m7, $m15, 16
	stm32 $m0, $m1, $m2
	ld32 $m0, $m7, $m15, 4
	add $m2, $m2, 1
	ld32 $m0, $m0, $m15, 0
	cmpult $m0, $m2, $m0
	ld32 $m1, $m11, $m15, 9                 # 4-byte Folded Reload
	ld32 $m4, $m11, $m15, 8                 # 4-byte Folded Reload
	add $m1, $m4, $m1
	st32 $m1, $m11, $m15, 9                 # 4-byte Folded Spill
	ld32 $m1, $m11, $m15, 15                # 4-byte Folded Reload
	ld32 $m4, $m11, $m15, 5                 # 4-byte Folded Reload
	add $m1, $m4, $m1
	st32 $m1, $m11, $m15, 15                # 4-byte Folded Spill
	brnz $m0, .LBB1_2
.LBB1_15:
	setzi $m0, 1
	ld32 $m7, $m11, $m15, 0                 # 4-byte Folded Reload
	ld32 $m10, $m11, $m15, 1                # 4-byte Folded Reload
	ld32 $m9, $m11, $m15, 2                 # 4-byte Folded Reload
	ld32 $m8, $m11, $m15, 3                 # 4-byte Folded Reload
	add $m11, $m11, 112
	br $m10
.Lfunc_end1:
	.size	_ZN8SWAffine7computeEv, .Lfunc_end1-_ZN8SWAffine7computeEv
	.section	.stack_sizes,"Go",@progbits,_ZN8SWAffine7computeEv,comdat,.text._ZN8SWAffine7computeEv
	.long	.Lfunc_begin1
	.byte	112
	.section	.text._ZN8SWAffine7computeEv,"axG",@progbits,_ZN8SWAffine7computeEv,comdat
                                        # -- End function
	.ident	"clang version 14.0.0 (ssh://git@phabricator.sourcevertex.net/diffusion/LLVMPROJECT/llvm-project.git 08ae98fc0ec57dd5363082ad6a63dbf031be762c)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
