#ifdef __IPU__
// #include "poplar/AvailableVTypes.h"
// #include "poplar/TileConstants.hpp"
// #include "popops/EncodingConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

#define SWASM __runCodelet_SWAsm___float

#define VOFF_C               0
#define VOFF_BG              1

#define VOFF_SIM_MATRIX      2
#define VOFF_SIM_WIDTH    	 3

#define VOFF_GAP_INIT     	 4
#define VOFF_GAP_EXT      	 5

#define VOFF_BUF_SIZE     	 6
#define VOFF_A            	 7
#define VOFF_Alen          	 8
#define VOFF_B            	 9
#define VOFF_Blen          	 10

#define VOFF_SCORE        	 11

#define STACK_SIZE (0)

// define variables
#define i m9
#define alen m8
#define c m7
#define bg m6
#define a m5
#define b m4
#define bi m3
#define simwidth m2

#define sscore a7
#define lastnogap a6
#define agap a5
#define prevnogap a4
#define gige a3
#define ge a2

.global SWASM
.type SWASM, @function
DEF_STACK_SIZE_OWN STACK_SIZE SWASM

DEF_STACK_SIZE_OWN 24 .text.printfd
.section .text.printfd
.align 4
printfd:
	add $sp, $sp, -24
	st32 $m10, $sp, $mzero, 6

	mov $a0, $sscore
	call $m10, __extendsfdf2

	setzi $m2, .printfd.str
	st32 $m1, $sp, $mzero, 3
	st32 $m0, $sp, $mzero, 2

	ldz8 $m0, $b, $mzero, 0
	st32 $m0, $sp, $mzero, 4
	ldz8 $m0, $a, $mzero, $i
	st32 $m0, $sp, $mzero, 5

	st32 $i, $sp, $mzero, 1
	st32 $m2, $sp, $mzero, 0
	call $m10, printf

	ld32 $m10, $sp, $mzero, 6
	add $sp, $sp, 24
	br $m10

.size .text.printf, . - .text.printf

.section .text.SWASM
.align 4
SWASM:
	add $sp, $mworker_base, -STACK_SIZE

// zero C and bG
	ld32 $c, $mzero, $mvertex_base, VOFF_C
	ld32 $bg, $mzero, $mvertex_base, VOFF_BG
	ld32 $m0, $mzero, $mvertex_base, VOFF_BUF_SIZE
	ld32 $m0, $mzero, $m0, 0

	add $m0, $m0, 1
	shrs $m0, $m0, 1

	rpt $m0, (loopInit_end - loopInit_start)/8 - 1;
loopInit_start:
	{
		st64step $azeros, $mzero, $c+=, 1
		fnop
	}
	{
		st64step $azeros, $mzero, $bg+=, 1
		fnop
	}
loopInit_end:
	ld32 $m0, $mzero, $mvertex_base, VOFF_Alen
	ld32 $alen, $mzero, $m0, 0

	ld32 $b, $mzero, $mvertex_base, VOFF_B

	setzi $sscore, 0

	ld32 $m0, $mzero, $mvertex_base, VOFF_GAP_EXT
	ld32 $ge, $m0, $mzero, 0  // ge

	ld32 $m0, $mzero, $mvertex_base, VOFF_Blen
	ld32 $i, $mzero, $m0, 0

	ld32 $simwidth, $mzero, $mvertex_base, VOFF_SIM_WIDTH
	ld32 $simwidth, $simwidth, $mzero, 0
	mul $simwidth, $simwidth, 4 // width of sim data type!!!

	brz $i, loopB_end

	sub $i, $i, 1
.align 8
loopB:
	{
		ld32 $a, $mzero, $mvertex_base, VOFF_A
		setzi $prevnogap, 0
	}
	{
		ld32 $bg, $mzero, $mvertex_base, VOFF_BG
		setzi $lastnogap, 0
	}

	ldz8step $bi, $mzero, $b+=, 1
	mul $bi, $bi, $simwidth
	ld32 $m0, $mzero, $mvertex_base, VOFF_SIM_MATRIX
	add $bi, $bi, $m0

	ld32 $m0, $mzero, $mvertex_base, VOFF_GAP_INIT
	ld32 $agap, $m0, $mzero, 0 // gi

	{
		ld32 $c, $mzero, $mvertex_base, VOFF_C
		f32add $gige, $agap, $ge // a2 = gi+ge
	}

// .align 8
	{
		rpt $alen, (loopA_end - loopA_start)/8 - 1;
		fnop
	}
loopA_start:
	{
		ldz8step $m0, $mzero, $a+=, 1 // m0 = a[j]
	 	f32add $agap, $agap, $ge
	}
	{
		ld32 $a0, $mzero, $bg, 0
		f32add $a1, $lastnogap, $gige
	}
	{
		ld32 $a1, $mzero, $c, 0
		f32max $agap, $agap, $a1 // agap calculation
	}
	{
		nop
		f32v2add $a0:1, $a0:1, $a2:3 // calc bg + ge and c[j] + gi + ge
	}
	{
		nop
		f32max $a0, $a0, $a1  // a0 = bgj
	}
	{
		st32step $a0, $mzero, $bg+=, 1 // save bgj
		f32max $lastnogap, $a0, $azero
	}
	{
		ld32 $a0, $bi, $mzero, $m0
		f32max $lastnogap, $lastnogap, $agap
	}
	{
		nop
		f32add $a0, $a0, $prevnogap
	}
	{
		ld32 $prevnogap, $mzero, $c, 0
		f32max $lastnogap, $lastnogap, $a0
	}
	{
		st32step $lastnogap, $mzero, $c+=, 1
		f32max $sscore, $sscore, $lastnogap
	}
loopA_end:

	brnzdec $i, loopB

loopB_end:
	ld32 $m0, $mzero, $mvertex_base, VOFF_SCORE
	f32toi32 $a0, $sscore
	st32 $a0, $mzero, $m0, 0
	exitz $m15

.size SWASM, . - SWASM 

.section .data
.printfd.str: .asciz "%d: %f %d %d\n"

#endif