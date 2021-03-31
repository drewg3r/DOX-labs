# Lab3: Conducting a three-factor experiment with using linear regression

**Objective:** To conduct a fractional three-factor experiment. Make a planning matrix, find the coefficients of the regression equation, conduct 3 statistical tests. 

## 31.03.2021 UPDATE:
Displaying time elapsed by uniformity of dispersion check
```
✅Cochran’s C test passed
Elapsed time: 0.000085040 seconds
```

## Test run
```
> python3 ./main.py 
DOX Lab3
╒══════╤══════╤══════╤══════╤══════╤══════╕
│   X1 │   X2 │   X3 │   Y1 │   Y2 │   Y3 │
╞══════╪══════╪══════╪══════╪══════╪══════╡
│   -5 │   10 │   10 │  231 │  212 │  216 │
├──────┼──────┼──────┼──────┼──────┼──────┤
│   -5 │   60 │   20 │  211 │  211 │  219 │
├──────┼──────┼──────┼──────┼──────┼──────┤
│   15 │   10 │   20 │  210 │  210 │  212 │
├──────┼──────┼──────┼──────┼──────┼──────┤
│   15 │   60 │   10 │  227 │  229 │  214 │
╘══════╧══════╧══════╧══════╧══════╧══════╛
Regression equation: y = 228.417 + 0.017x1 + 0.067x2 + -0.933x3

Cochran test:
Dispersions:
d1 = 66.889
d2 = 14.222
d3 = 0.889
d4 = 44.222

gp = 0.530
✅Cochran’s C test passed
Elapsed time: 0.000084959 seconds

Student's t-test
Sb = 1.622
Beta:
b0 = 216.833
b1 = 0.167
b2 = 1.667
b3 = -4.667

t:
t0 = 133.715
t1 = 0.103
t2 = 1.028
t3 = 2.878

t1,t2 < t_tabl(t_tabl=2.306)
Factors b2,b3 can be excluded
Regression equation without excluded factors:
y = 228.417 + -0.933*x3 

y1 = 219.083
y2 = 209.750
y3 = 209.750
y4 = 219.083

F-test
d = 2
S2ad = 51.875
Fp = 1.644
✅F-test passed/model is adequate
```
