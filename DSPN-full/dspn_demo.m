function dspn_demo

display('training:');
dspn_train;
display(' ');

display('precalc. results:');
dspn_results('out_precalc/');
display(' ');

display('recalc. results:');
dspn_results;
