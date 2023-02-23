# Test Values

This file contains the correct results and can be used to validate PASCal if things change.

## Strain

### VT Input 
#Variable Temperature Data for Ag3CoCN6 Phase II
95      5       6.6200  11.3500 6.6230  90.0000 78.5000 90.0000
120     5       6.6310  11.3622 6.6140  90.0000 78.4680 90.0000
145     5       6.6328  11.3643 6.6099  90.0000 78.4700 90.0000
170     5       6.6361  11.3759 6.6007  90.0000 78.4330 90.0000
195     5       6.6330  11.3829 6.6062  90.0000 78.3690 90.0000
220     5       6.6379  11.3883 6.6000  90.0000 78.3790 90.0000
245     5       6.6383  11.3969 6.5980  90.0000 78.3720 90.0000
270     5       6.6401  11.4091 6.5956  90.0000 78.3490 90.0000
300     5       6.6415  11.4251 6.5959  90.0000 78.2840 90.0000
                  
### PASCal, VT, Euler, Finite
T (K)	X1 (%)	X2 (%)	X3 (%)	X1,calc (%)	X2,calc (%)	X3,calc (%)
95.0	0.0	0.0	0.0	-0.0858	-0.0044	0.0919
120.0	-0.1474	0.1074	0.1661	-0.1424	0.0707	0.1246
145.0	-0.2091	0.1259	0.1937	-0.1991	0.1458	0.1572
170.0	-0.361	0.2279	0.243	-0.2557	0.2209	0.1899
195.0	-0.3097	0.2894	0.2056	-0.3123	0.296	0.2225
220.0	-0.3935	0.3369	0.273	-0.3689	0.3711	0.2551
245.0	-0.4264	0.4124	0.2792	-0.4255	0.4461	0.2878
270.0	-0.4725	0.5193	0.308	-0.4821	0.5212	0.3204
300.0	-0.5022	0.6595	0.3404	-0.55	0.6113	0.3596

0.00664 0.00341 -0.00500

### PASCal VT Lagrangian Finite

T (K)	X1 (%)	X2 (%)	X3 (%)	X1,calc (%)	X2,calc (%)	X3,calc (%)
95.0	0.0	0.0	0.0	-0.0858	-0.005	0.0921
120.0	-0.1472	0.1075	0.1663	-0.1422	0.0706	0.1248
145.0	-0.2087	0.1261	0.194	-0.1986	0.1461	0.1575
170.0	-0.3598	0.2285	0.2435	-0.2549	0.2217	0.1902
195.0	-0.309	0.2903	0.2058	-0.3113	0.2973	0.2228
220.0	-0.3922	0.338	0.2735	-0.3677	0.3728	0.2555
245.0	-0.4249	0.4141	0.2797	-0.424	0.4484	0.2882
270.0	-0.4707	0.5221	0.3085	-0.4804	0.5239	0.3209
300.0	-0.5004	0.6639	0.3409	-0.5481	0.6146	0.3601


### Bilbao VT Lagrangian Finite
120 0.00108 0.00166 -0.00147
145 0.00126 0.00194 -0.00209
170 0.00228 0.00243 -0.00360
195 0.00290 0.00206 -0.00309
220 0.00338 0.00273 -0.00392
245 0.00414 0.00280 -0.00425
270 0.00522 0.00308 -0.00471
300 0.00664 0.00341 -0.00500

### Bilbao VT Lagrangian Infinite
120 0.00107 0.00166 -0.00147
145 0.00126 0.00194 -0.00209
170 0.00228 0.00243 -0.00361
195 0.00290 0.00205 -0.00310
220 0.00337 0.00273 -0.00393
245 0.00413 0.00279 -0.00426
270 0.00521 0.00308 -0.00472
300 0.00662 0.00340 -0.00502

### VP Input
0.190	1	6.6985	11.5782	6.5554	90	78.522	90
0.226	1	6.6934	11.5391	6.5661	90	78.521	90.0000
0.290	1	6.68	11.468	6.577	90	78.47	90.0000
0.294	1	6.6745	11.497	6.5707	90	78.473	90.0000
0.395	1	6.6415	11.4251	6.5959	90	78.284	90.0000
0.55	1	6.605	11.295	6.629	90	78.21	90.0000
0.78	1	6.553	11.196	6.646	90	77.967	90.0000
0.90	1	6.539	11.156	6.654	90	77.871	90.0000
0.92	1	6.5333	11.154	6.6507	90	77.91	90.0000
1.08	1	6.505	11.116	6.663	90	77.73	90.0000
1.87	1	6.4172	10.922	6.696	90	77.584	90.0000
3.02	1	6.2904	10.739	6.747	90	77.39	90.0000
5.13	1	6.0657	10.493	6.833	90	77.09	90.0000
6.92	1	5.91	10.333	6.891	90	77.01	90.0000
7.65	1	5.8657	10.27	6.908	90	76.921	90.0000

### PASCAl VP Lag Fin
P (GPa)	X1 (%)	X2 (%)	X3 (%)	X1,calc (%)	X2,calc (%)	X3,calc (%)
0.19	-0.0	0.0	0.0	0.0573	0.1279	-0.0281
0.226	-0.3371	-0.0787	0.1656	-0.316	-0.122	0.1899
0.29	-0.9473	-0.2946	0.3305	-0.8526	-0.4745	0.4036
0.294	-0.6989	-0.3754	0.2342	-0.8822	-0.4942	0.4147
0.395	-1.3136	-0.9331	0.6218	-1.5306	-0.9378	0.6502
0.55	-2.4161	-1.495	1.1291	-2.2867	-1.497	0.9277
0.78	-3.2465	-2.3443	1.3951	-3.1315	-2.1877	1.2553
0.9	-3.58	-2.5844	1.5218	-3.4941	-2.5077	1.4035
0.92	-3.5967	-2.6508	1.4681	-3.5507	-2.559	1.427
1.08	-3.9123	-3.129	1.6644	-3.9718	-2.9523	1.6064
1.87	-5.5069	-4.4308	2.1708	-5.5309	-4.595	2.336
3.02	-6.9854	-6.277	2.9655	-7.0641	-6.516	3.1622
5.13	-8.9335	-9.4452	4.3297	-8.9795	-9.3704	4.3557
6.92	-10.1764	-11.5354	5.2707	-10.1778	-11.4257	5.1972
7.65	-10.6605	-12.1382	5.5452	-10.5999	-12.2005	5.5112

## Bilbao VP Lag Fin
7.65 -0.10661 -0.12138 0.05545


### PASCal VP Lag Inf
P (GPa)	X1 (%)	X2 (%)	X3 (%)	X1,calc (%)	X2,calc (%)	X3,calc (%)
0.19	-0.0	0.0	0.0	0.0615	0.1223	-0.0311
0.226	-0.3377	-0.0788	0.1655	-0.3177	-0.1335	0.1861
0.29	-0.9518	-0.295	0.3299	-0.8603	-0.4835	0.4018
0.294	-0.7013	-0.3761	0.2339	-0.8903	-0.503	0.4129
0.395	-1.3223	-0.9375	0.6199	-1.546	-0.9425	0.6492
0.55	-2.446	-1.5064	1.1228	-2.3145	-1.5013	0.9257
0.78	-3.301	-2.3725	1.3854	-3.1803	-2.2004	1.2501
0.9	-3.6465	-2.6188	1.5103	-3.5547	-2.5277	1.3962
0.92	-3.6638	-2.6869	1.4574	-3.6133	-2.5803	1.4195
1.08	-3.992	-3.1797	1.6506	-4.0505	-2.9853	1.5958
1.87	-5.6675	-4.5336	2.1477	-5.6899	-4.7043	2.3093
3.02	-7.2481	-6.4875	2.9228	-7.3331	-6.7593	3.111
5.13	-9.3728	-9.9395	4.2395	-9.4266	-9.881	4.2607
6.92	-10.7547	-12.2923	5.1371	-10.758	-12.1688	5.0666
7.65	-11.2988	-12.9825	5.3976	-11.2306	-13.0386	5.3666


## Bilbao VP Lag Inf
7.65 -0.11299 -0.12982 0.05398