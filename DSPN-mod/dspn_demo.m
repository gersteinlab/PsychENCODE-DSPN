function dspn_demo

display('training:');
dspn_train;
display(' ');

display('precalc. results (calculated on MATLAB_R2017a):');
dspn_results('out_precalc/');
display(' ');

display('recalc. results:');
dspn_results;
