x = linspace(0,1,101);
x = x';
y = [
0.
0.00318284
0.00636253
0.00953595
0.0127
0.0158514
0.0189873
0.0221044
0.0251996
0.02827
0.0313125
0.0343242
0.0373019
0.0402428
0.043144
0.0460026
0.0488159
0.0515809
0.0542951
0.0569556
0.05956
0.0621056
0.0645899
0.0670104
0.0693648
0.0716508
0.073866
0.0760084
0.0780757
0.080066
0.0819773
0.0838077
0.0855553
0.0872186
0.0887957
0.0902853
0.0916857
0.0929956
0.0942138
0.095339
0.0963701
0.0973061
0.0981461
0.0988892
0.0995347
0.100082
0.100531
0.10088
0.10113
0.10128
0.10133
0.10128
0.10113
0.10088
0.100531
0.100082
0.0995347
0.0988892
0.0981461
0.0973061
0.0963701
0.095339
0.0942138
0.0929956
0.0916857
0.0902853
0.0887957
0.0872186
0.0855553
0.0838077
0.0819773
0.080066
0.0780757
0.0760084
0.073866
0.0716508
0.0693648
0.0670104
0.0645899
0.0621056
0.05956
0.0569556
0.0542951
0.0515809
0.0488159
0.0460026
0.043144
0.0402428
0.0373019
0.0343242
0.0313125
0.02827
0.0251996
0.0221044
0.0189873
0.0158514
0.0127
0.00953595
0.00636253
0.00318284
0.
];
y1=[
    0.
0.00379567
0.00700064
0.0101962
0.0133792
0.0165466
0.0196952
0.022822
0.025924
0.0289981
0.0320412
0.0350505
0.038023
0.0409558
0.0438459
0.0466907
0.0494873
0.052233
0.0549251
0.0575611
0.0601382
0.062654
0.0651061
0.0674921
0.0698096
0.0720564
0.0742303
0.0763292
0.0783511
0.0802939
0.0821558
0.083935
0.0856298
0.0872385
0.0887596
0.0901916
0.0915331
0.0927828
0.0939396
0.0950022
0.0959698
0.0968413
0.097616
0.098293
0.0988719
0.0993519
0.0997328
0.100014
0.100196
0.100277
0.100259
0.10014
0.0999221
0.0996044
0.0991874
0.0986716
0.0980577
0.0973461
0.0965377
0.0956333
0.0946337
0.0935401
0.0923534
0.0910751
0.0897062
0.0882482
0.0867026
0.0850709
0.0833548
0.081556
0.0796762
0.0777175
0.0756817
0.0735709
0.0713872
0.0691328
0.0668099
0.0644209
0.0619682
0.0594542
0.0568814
0.0542524
0.0515698
0.0488362
0.0460546
0.0432275
0.0403578
0.0374485
0.0345023
0.0315222
0.0285113
0.0254724
0.0224087
0.0193231
0.0162189
0.013099
0.0099665
0.00682464
0.0036765
0.
];
plot(x,y);
hold on;
% plot(x,y1);
exdata(:,1) = x;
exdata(:,2) = y;

% imdata(:,1) = x;
% imdata(:,2) = y1;
writematrix(exdata,'explicit_CFL0.4.dat');
% writematrix(imdata,'implicit.dat');