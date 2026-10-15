# Lossless MoE study: tables

## BF16

### Stage A: stored bytes / BF16 bytes (same bytes for every codec)

| Model | Matrices | Weights (M) | zstd-3 | zstd-19 | zstd-19 byte split | zstd-19 planes | nvCOMP ANS | lmc order-0 (15-bit) | lmc order-0 (12-bit) | lmc previous exponent | lmc column class | lmc row class | lmc order-0 bits/weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| granite | 96 | 50.3 | 0.7798 | 0.7773 | 0.6713 | 0.6650 | 0.6713 | 0.6626 | 0.6628 | 0.6744 | 0.6702 | 0.6693 | 10.602 |
| moonlight | 144 | 415.2 | 0.7828 | 0.7762 | 0.6724 | 0.6610 | 0.6742 | 0.6594 | 0.6595 | 0.6617 | 0.6610 | 0.6608 | 10.550 |
| olmoe | 96 | 201.3 | 0.7781 | 0.7729 | 0.6686 | 0.6623 | 0.6692 | 0.6603 | 0.6604 | 0.6634 | 0.6623 | 0.6620 | 10.564 |

Subset (the first sampled expert of each layer): LZMA preset 9e, bzip2 9, zstd 22 against lmc order-0 on the same matrices.

| Model | LZMA 9e | LZMA 9e byte split | bzip2 9 | zstd 22 | zstd-19 planes | lmc order-0 |
| --- | --- | --- | --- | --- | --- | --- |
| granite | 0.7234 | 0.6867 | 0.6969 | 0.7773 | 0.6647 | 0.6631 |
| moonlight | 0.7126 | 0.6877 | 0.6898 | 0.7764 | 0.6609 | 0.6592 |
| olmoe | 0.7142 | 0.6840 | 0.6935 | 0.7725 | 0.6622 | 0.6601 |

### lmc decode throughput (GB/s of BF16 output, best of repeats, decode tables built in the timed region)

| Variant | 1 thread | 6 threads | 12 threads | peak working set (GB) |
| --- | --- | --- | --- | --- |
| order0 | 0.52 | 2.87 | 2.88 | 2.25 |
| order0_s12 | 0.60 | 3.28 | 3.28 | 2.25 |
| left | 0.40 | 2.09 | 2.09 | 2.28 |
| col16 | 0.38 | 2.07 | 2.08 | 2.25 |
| row16 | 0.43 | 2.37 | 2.37 | 2.25 |
| trained | 0.53 | 2.88 | 2.86 | 1.13 |

Python codecs, single thread (GB/s of BF16 out): zstd-3/raw 0.72, zstd-19/raw 0.44, zstd-19/byte_split 1.72, zstd-19/planes 1.76

### Stage B: bits per weight beyond order-0, matched matrices

| Model | Field | Order-0 (adaptive) | Context | In-sample | Adaptive | Two-part |
| --- | --- | --- | --- | --- | --- | --- |
| granite | exponent | 2.5905 | row_class16 | -0.0152 | -0.0113 | -0.0041 |
| granite | exponent | 2.5905 | row_mean64 | -0.0043 | -0.0039 | -0.0020 |
| granite | exponent | 2.5905 | neuron_class16 | -0.0073 | -0.0033 | +0.0040 |
| granite | mantissa | 6.9736 | exponent | -0.0493 | -0.0374 | +0.0420 |
| granite | mantissa_top3 | 2.9731 | exponent | -0.0455 | -0.0443 | -0.0398 |
| granite | mantissa_top3 | 2.9731 | exponent+col_class16 | -0.0416 | -0.0300 | +0.0313 |
| granite | sign | 1.0000 | above | -0.0001 | -0.0001 | +0.0000 |
| granite | sign | 1.0000 | given_gate | -0.0000 | -0.0000 | +0.0000 |
| granite | sign | 1.0000 | given_prev | -0.0000 | +0.0000 | +0.0001 |
| moonlight | exponent | 2.5474 | row_mean64 | -0.0011 | -0.0009 | -0.0001 |
| moonlight | exponent | 2.5474 | row_class16 | -0.0015 | -0.0006 | +0.0006 |
| moonlight | exponent | 2.5474 | row_mean8 | -0.0002 | -0.0001 | +0.0007 |
| moonlight | mantissa | 6.9725 | exponent | -0.0502 | -0.0473 | -0.0333 |
| moonlight | mantissa_top3 | 2.9730 | exponent | -0.0486 | -0.0484 | -0.0476 |
| moonlight | mantissa_top3 | 2.9730 | exponent+col_class16 | -0.0465 | -0.0436 | -0.0321 |
| moonlight | sign | 1.0000 | above | -0.0000 | -0.0000 | +0.0000 |
| moonlight | sign | 1.0000 | given_gate | -0.0000 | +0.0000 | +0.0000 |
| moonlight | sign | 1.0000 | given_prev | -0.0000 | +0.0000 | +0.0000 |
| olmoe | exponent | 2.5609 | row_class16 | -0.0050 | -0.0038 | -0.0021 |
| olmoe | exponent | 2.5609 | col_mean_causal | -0.0006 | -0.0004 | +0.0007 |
| olmoe | exponent | 2.5609 | row_mean64 | -0.0005 | -0.0003 | +0.0007 |
| olmoe | mantissa | 6.9726 | exponent | -0.0505 | -0.0467 | -0.0267 |
| olmoe | mantissa_top3 | 2.9730 | exponent | -0.0487 | -0.0484 | -0.0473 |
| olmoe | mantissa_top3 | 2.9730 | exponent+col_class16 | -0.0464 | -0.0426 | -0.0267 |
| olmoe | sign | 1.0000 | above | -0.0000 | -0.0000 | +0.0000 |
| olmoe | sign | 1.0000 | given_prev | -0.0000 | +0.0000 | +0.0000 |
| olmoe | sign | 1.0000 | given_gate | -0.0000 | +0.0000 | +0.0000 |

Held-out (tables from calibration matrices of the same kind, scored on evaluation matrices; prior-KT, strength 256): order-0 and the best context per group.

| Group | Order-0 | Best context | Delta |
| --- | --- | --- | --- |
| granite/down/experts | 2.5834 | col_class16+row_mean64 | -0.0037 |
| granite/down/layers | 2.5890 | col_class16+row_mean64 | -0.0084 |
| granite/gate/experts | 2.5891 | row_class16 | -0.0083 |
| granite/gate/layers | 2.5951 | row_class16 | -0.0080 |
| granite/up/experts | 2.5928 | row_class16 | -0.0142 |
| granite/up/layers | 2.5826 | row_class16 | -0.0067 |
| moonlight/down/experts | 2.5514 | row_mean64 | -0.0013 |
| moonlight/down/layers | 2.5554 | col_mean_causal | -0.0036 |
| moonlight/gate/experts | 2.5575 | row_class16 | -0.0047 |
| moonlight/gate/layers | 2.5609 | row_class16 | -0.0064 |
| moonlight/up/experts | 2.5488 | row_mean64 | -0.0006 |
| moonlight/up/layers | 2.5494 | col_mean_causal | -0.0007 |
| olmoe/down/experts | 2.5630 | col_class16+row_mean64 | -0.0021 |
| olmoe/down/layers | 2.5607 | col_class16+row_mean64 | -0.0026 |
| olmoe/gate/experts | 2.5771 | row_class16 | -0.0109 |
| olmoe/gate/layers | 2.5754 | row_class16 | -0.0161 |
| olmoe/up/experts | 2.5653 | row_class16 | -0.0027 |
| olmoe/up/layers | 2.5628 | row_class16 | -0.0041 |

## Quantized experts (fetched sample)

### Stage A: stored bytes / stored bytes of the container (weights plus scales)

| Source | Format | Matrices | Weights (M) | Scales share | zstd-19 | nvCOMP ANS | Colibri g0 | Colibri per-matrix | lmc order-0 | lmc byte pairs | lmc best context | lmc order-0 bits/weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| glm52_int4_row | int4_row | 48 | 604.0 | 0.0022 | 0.7460 | 0.7588 | 0.7376 | 0.7372 | 0.7369 | 0.7398 | 0.7355 (row16) | 2.9540 |
| gptoss20b_mxfp4 | mxfp4 | 32 | 398.1 | 0.0588 | 0.9199 | 0.9722 | 0.9681 | 0.9677 | 0.9221 | 0.9204 | 0.9085 (scale16) | 3.9189 |
| inkling_int4_row | int4_row | 48 | 906.0 | 0.0017 | 0.7421 | 0.7547 | 0.7328 | 0.7326 | 0.7324 | 0.7355 | 0.7309 (row16) | 2.9348 |
| kimi_k2_int4_g32 | int4_g32_bf16 | 48 | 704.6 | 0.1111 | 0.8665 | 0.9224 | 0.9163 | 0.9163 | 0.8668 | 0.8668 | 0.8629 (scale16) | 3.9006 |
| mimo_v26_pro_mxfp4 | mxfp4 | 48 | 604.0 | 0.0588 | 0.8880 | 0.9440 | 0.9478 | 0.9384 | 0.8865 | 0.8860 | 0.8715 (col16) | 3.7674 |

### Decode throughput (GB/s of stored weight bytes out, best of repeats)

| Variant | Threads | lmc | Colibri scalar | Colibri branch-free |
| --- | --- | --- | --- | --- |
| colibri_g0 | 1 | 0.118 | 0.095 | 0.086 |
| colibri_g0 | 6 | 0.757 | 0.521 | 0.469 |
| colibri_g0 | 12 | 0.806 | 0.523 | 0.469 |
| colibri_tensor | 1 | 0.156 | 0.095 | 0.086 |
| colibri_tensor | 6 | 0.811 | 0.520 | 0.469 |
| colibri_tensor | 12 | 0.818 | 0.522 | 0.470 |
| order0 | 1 | 0.156 |  |  |
| order0 | 6 | 0.821 |  |  |
| order0 | 12 | 0.822 |  |  |
| pairs | 1 | 0.354 |  |  |
| pairs | 6 | 1.909 |  |  |
| pairs | 12 | 1.918 |  |  |
| left | 1 | 0.115 |  |  |
| left | 6 | 0.623 |  |  |
| left | 12 | 0.624 |  |  |
| col16 | 1 | 0.108 |  |  |
| col16 | 6 | 0.579 |  |  |
| col16 | 12 | 0.585 |  |  |
| row16 | 1 | 0.127 |  |  |
| row16 | 6 | 0.671 |  |  |
| row16 | 12 | 0.678 |  |  |
| scale16 | 1 | 0.110 |  |  |
| scale16 | 6 | 0.580 |  |  |
| scale16 | 12 | 0.581 |  |  |
| scales_order0 | 1 | 0.487 |  |  |
| scales_order0 | 6 | 2.586 |  |  |
| scales_order0 | 12 | 2.617 |  |  |
| scales_left | 1 | 0.259 |  |  |
| scales_left | 6 | 1.328 |  |  |
| scales_left | 12 | 1.315 |  |  |

### Stage B: bits per weight beyond order-0, matched matrices

| Source | Field | Order-0 (adaptive) | Context | In-sample | Adaptive | Two-part |
| --- | --- | --- | --- | --- | --- | --- |
| glm52_int4_row | code | 2.9452 | row_class16 | -0.0070 | -0.0068 | -0.0067 |
| glm52_int4_row | code | 2.9452 | scale_class16 | -0.0056 | -0.0056 | -0.0055 |
| glm52_int4_row | code | 2.9452 | row_mean512 | -0.0052 | -0.0051 | -0.0048 |
| glm52_int4_row | code | 2.9452 | scale_class16+col_class16 | -0.0052 | -0.0046 | -0.0036 |
| gptoss20b_mxfp4 | code | 3.8614 | scale_class16+col_class16 | -0.0923 | -0.0916 | -0.0891 |
| gptoss20b_mxfp4 | code | 3.8614 | block_max+scale_class16 | -0.0638 | -0.0632 | -0.0609 |
| gptoss20b_mxfp4 | code | 3.8614 | scale_class16 | -0.0580 | -0.0579 | -0.0578 |
| gptoss20b_mxfp4 | code | 3.8614 | col_mean_causal | -0.0324 | -0.0323 | -0.0321 |
| gptoss20b_mxfp4 | scale_e8m0 | 1.8193 | row_class16 | -0.9056 | -0.9033 | -0.8992 |
| gptoss20b_mxfp4 | scale_e8m0 | 1.8193 | left | -0.6646 | -0.6635 | -0.6600 |
| gptoss20b_mxfp4 | scale_e8m0 | 1.8193 | above | -0.3054 | -0.3043 | -0.3008 |
| gptoss20b_mxfp4 | scale_e8m0 | 1.8193 | col_class16 | -0.0048 | -0.0026 | +0.0016 |
| inkling_int4_row | code | 2.9277 | rank1_class32 | -0.0077 | -0.0075 | -0.0073 |
| inkling_int4_row | code | 2.9277 | row_class16 | -0.0074 | -0.0073 | -0.0072 |
| inkling_int4_row | code | 2.9277 | col_class16+row_mean512 | -0.0074 | -0.0068 | -0.0049 |
| inkling_int4_row | code | 2.9277 | scale_class16+col_class16 | -0.0069 | -0.0063 | -0.0052 |
| kimi_k2_int4_g32 | code | 3.6223 | block_max+scale_class16 | -0.0360 | -0.0356 | -0.0344 |
| kimi_k2_int4_g32 | code | 3.6223 | scale_class16 | -0.0177 | -0.0177 | -0.0176 |
| kimi_k2_int4_g32 | code | 3.6223 | scale_class16+col_class16 | -0.0171 | -0.0166 | -0.0154 |
| kimi_k2_int4_g32 | code | 3.6223 | block_max | -0.0166 | -0.0165 | -0.0164 |
| kimi_k2_int4_g32 | scale_bf16_exponent | 0.8873 | left | -0.0007 | -0.0005 | -0.0001 |
| kimi_k2_int4_g32 | scale_bf16_exponent | 0.8873 | above | -0.0000 | +0.0001 | +0.0006 |
| kimi_k2_int4_g32 | scale_bf16_exponent | 0.8873 | col_class16 | +0.0020 | +0.0027 | +0.0041 |
| kimi_k2_int4_g32 | scale_bf16_exponent | 0.8873 | row_class16 | +0.0185 | +0.0192 | +0.0205 |
| kimi_k2_int4_g32 | scale_bf16_mantissa | 6.6819 | exponent | -0.6198 | -0.6177 | -0.6073 |
| mimo_v26_pro_mxfp4 | code | 3.7370 | scale_class16+col_class16 | -0.1132 | -0.1126 | -0.1111 |
| mimo_v26_pro_mxfp4 | code | 3.7370 | col_class16+row_mean512 | -0.0805 | -0.0794 | -0.0761 |
| mimo_v26_pro_mxfp4 | code | 3.7370 | rank1_class32 | -0.0772 | -0.0769 | -0.0766 |
| mimo_v26_pro_mxfp4 | code | 3.7370 | col_mean_causal | -0.0660 | -0.0659 | -0.0657 |
| mimo_v26_pro_mxfp4 | scale_e8m0 | 0.9501 | row_class16 | -0.1260 | -0.1250 | -0.1230 |
| mimo_v26_pro_mxfp4 | scale_e8m0 | 0.9501 | left | -0.0749 | -0.0746 | -0.0736 |
| mimo_v26_pro_mxfp4 | scale_e8m0 | 0.9501 | col_class16 | -0.0249 | -0.0238 | -0.0217 |
| mimo_v26_pro_mxfp4 | scale_e8m0 | 0.9501 | above | -0.0071 | -0.0068 | -0.0058 |

Held-out (prior-KT 256): order-0 and the best context per group.

| Group | Order-0 | Best context | Delta |
| --- | --- | --- | --- |
| glm52_int4_row/down/experts | 3.0140 | row_class16 | -0.0079 |
| glm52_int4_row/down/layers | 3.0093 | scale_class16+col_class16 | -0.0083 |
| glm52_int4_row/gate/experts | 2.9111 | row_class16 | -0.0065 |
| glm52_int4_row/gate/layers | 2.9135 | row_class16 | -0.0065 |
| glm52_int4_row/up/experts | 2.9115 | row_class16 | -0.0066 |
| glm52_int4_row/up/layers | 2.9150 | row_class16 | -0.0065 |
| gptoss20b_mxfp4/down/experts | 3.8543 | scale_class16+col_class16 | -0.0939 |
| gptoss20b_mxfp4/down/layers | 3.8491 | scale_class16+col_class16 | -0.1053 |
| gptoss20b_mxfp4/gate_up/experts | 3.8657 | scale_class16+col_class16 | -0.0730 |
| gptoss20b_mxfp4/gate_up/layers | 3.8650 | scale_class16+col_class16 | -0.0778 |
| inkling_int4_row/down/experts | 2.9656 | rank1_class32 | -0.0098 |
| inkling_int4_row/down/layers | 2.9727 | row_class16 | -0.0078 |
| inkling_int4_row/gate/experts | 2.9139 | row_class16 | -0.0070 |
| inkling_int4_row/gate/layers | 2.9173 | row_class16 | -0.0070 |
| inkling_int4_row/up/experts | 2.9151 | row_class16 | -0.0069 |
| inkling_int4_row/up/layers | 2.9185 | row_class16 | -0.0068 |
| kimi_k2_int4_g32/down/experts | 3.6190 | block_max+scale_class16 | -0.0364 |
| kimi_k2_int4_g32/down/layers | 3.6192 | block_max+scale_class16 | -0.0364 |
| kimi_k2_int4_g32/gate/experts | 3.6230 | block_max+scale_class16 | -0.0358 |
| kimi_k2_int4_g32/gate/layers | 3.6232 | block_max+scale_class16 | -0.0358 |
| kimi_k2_int4_g32/up/experts | 3.6230 | block_max+scale_class16 | -0.0359 |
| kimi_k2_int4_g32/up/layers | 3.6231 | block_max+scale_class16 | -0.0359 |
| mimo_v26_pro_mxfp4/down/experts | 3.7894 | col_mean_causal | -0.0697 |
| mimo_v26_pro_mxfp4/down/layers | 3.8166 | scale_class16+col_class16 | -0.0587 |
| mimo_v26_pro_mxfp4/gate/experts | 3.7257 | col_class16+row_mean512 | -0.2395 |
| mimo_v26_pro_mxfp4/gate/layers | 3.8087 | block_max+scale_class16 | -0.0402 |
| mimo_v26_pro_mxfp4/up/experts | 3.7607 | col_class16+row_mean512 | -0.1492 |
| mimo_v26_pro_mxfp4/up/layers | 3.8200 | block_max+scale_class16 | -0.0527 |

## Diagnostic: Moonlight re-quantized per-row int4 (not a principal result)

### Stage A: stored bytes / stored bytes of the container (weights plus scales)

| Source | Format | Matrices | Weights (M) | Scales share | zstd-19 | nvCOMP ANS | Colibri g0 | Colibri per-matrix | lmc order-0 | lmc byte pairs | lmc best context | lmc order-0 bits/weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| moonlight_requant_int4 | int4_row | 144 | 415.2 | 0.0045 | 0.7668 | 0.7808 | 0.7615 | 0.7615 | 0.7603 | 0.7627 | 0.7588 (scale16) | 3.0549 |

### Decode throughput (GB/s of stored weight bytes out, best of repeats)

| Variant | Threads | lmc | Colibri scalar | Colibri branch-free |
| --- | --- | --- | --- | --- |
| colibri_g0 | 1 | 0.157 | 0.093 | 0.082 |
| colibri_g0 | 6 | 0.843 | 0.512 | 0.451 |
| colibri_g0 | 12 | 0.841 | 0.511 | 0.448 |
| colibri_tensor | 1 | 0.156 | 0.093 | 0.082 |
| colibri_tensor | 6 | 0.842 | 0.512 | 0.450 |
| colibri_tensor | 12 | 0.841 | 0.510 | 0.446 |
| order0 | 1 | 0.157 |  |  |
| order0 | 6 | 0.836 |  |  |
| order0 | 12 | 0.841 |  |  |
| pairs | 1 | 0.350 |  |  |
| pairs | 6 | 1.878 |  |  |
| pairs | 12 | 1.887 |  |  |
| left | 1 | 0.125 |  |  |
| left | 6 | 0.669 |  |  |
| left | 12 | 0.670 |  |  |
| col16 | 1 | 0.111 |  |  |
| col16 | 6 | 0.592 |  |  |
| col16 | 12 | 0.589 |  |  |
| row16 | 1 | 0.127 |  |  |
| row16 | 6 | 0.685 |  |  |
| row16 | 12 | 0.680 |  |  |
| scale16 | 1 | 0.110 |  |  |
| scale16 | 6 | 0.592 |  |  |
| scale16 | 12 | 0.593 |  |  |

### Stage B: bits per weight beyond order-0, matched matrices

| Source | Field | Order-0 (adaptive) | Context | In-sample | Adaptive | Two-part |
| --- | --- | --- | --- | --- | --- | --- |
| moonlight_requant_int4 | code | 3.0361 | row_class16 | -0.0077 | -0.0071 | -0.0064 |
| moonlight_requant_int4 | code | 3.0361 | scale_class16 | -0.0067 | -0.0065 | -0.0063 |
| moonlight_requant_int4 | code | 3.0361 | row_mean512 | -0.0066 | -0.0063 | -0.0053 |
| moonlight_requant_int4 | code | 3.0361 | row_mean64 | -0.0039 | -0.0036 | -0.0026 |

Held-out (prior-KT 256): order-0 and the best context per group.

| Group | Order-0 | Best context | Delta |
| --- | --- | --- | --- |
| moonlight_requant_int4/down/experts | 3.0612 | scale_class16 | -0.0081 |
| moonlight_requant_int4/down/layers | 3.0599 | scale_class16 | -0.0083 |
| moonlight_requant_int4/gate/experts | 3.0228 | row_class16 | -0.0075 |
| moonlight_requant_int4/gate/layers | 3.0200 | row_class16 | -0.0078 |
| moonlight_requant_int4/up/experts | 3.0237 | row_class16 | -0.0075 |
| moonlight_requant_int4/up/layers | 3.0205 | row_class16 | -0.0078 |

## Gate (pre-registered)

Best achievable reduction against the strongest practical static rANS coder (lmc order-0 with every header, table and stream byte, the scales coded order-0 where they are compressible), on the same matrices. Every matrix and field takes its own best context in hindsight. Achievable: adaptive code lengths (all learning costs included). Optimistic: in-sample conditional entropies (free tables fitted to the data; biased low).

| Corpus | Static rANS (bits/weight) | Achievable gain | Optimistic gain | Achievable reduction | Optimistic reduction | Stage B verdict |
| --- | --- | --- | --- | --- | --- | --- |
| glm52_int4_row | 2.9540 | 0.0072 | 0.0073 | 0.24% | 0.25% | unsupported |
| gptoss20b_mxfp4 | 3.9189 | 0.1198 | 0.1206 | 3.06% | 3.08% | unsupported |
| inkling_int4_row | 2.9348 | 0.0092 | 0.0093 | 0.31% | 0.32% | unsupported |
| kimi_k2_int4_g32 | 3.9006 | 0.0550 | 0.0555 | 1.41% | 1.42% | unsupported |
| mimo_v26_pro_mxfp4 | 3.7674 | 0.1260 | 0.1265 | 3.34% | 3.36% | unsupported |
| granite | 10.6021 | 0.0534 | 0.0697 | 0.50% | 0.66% | unsupported |
| moonlight | 10.5497 | 0.0497 | 0.0535 | 0.47% | 0.51% | unsupported |
| olmoe | 10.5640 | 0.0528 | 0.0577 | 0.50% | 0.55% | unsupported |

### Diagnostic gate (not a principal result)

Best achievable reduction against the strongest practical static rANS coder (lmc order-0 with every header, table and stream byte, the scales coded order-0 where they are compressible), on the same matrices. Every matrix and field takes its own best context in hindsight. Achievable: adaptive code lengths (all learning costs included). Optimistic: in-sample conditional entropies (free tables fitted to the data; biased low).

| Corpus | Static rANS (bits/weight) | Achievable gain | Optimistic gain | Achievable reduction | Optimistic reduction | Stage B verdict |
| --- | --- | --- | --- | --- | --- | --- |
| moonlight_requant_int4 | 3.0549 | 0.0074 | 0.0079 | 0.24% | 0.26% | unsupported |

