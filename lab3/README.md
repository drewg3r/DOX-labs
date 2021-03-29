# Lab3: Conducting a three-factor experiment with using linear regression

**Objective:** To conduct a fractional three-factor experiment. Make a planning matrix, find the coefficients of the regression equation, conduct 3 statistical tests. 

## Test run
```
> python3 ./main.py 
DOX Lab3
╒══════╤══════╤══════╤══════╤══════╤══════╕
│   X1 │   X2 │   X3 │   Y1 │   Y2 │   Y3 │
╞══════╪══════╪══════╪══════╪══════╪══════╡
│   -5 │   10 │   10 │  214 │  231 │  206 │
├──────┼──────┼──────┼──────┼──────┼──────┤
│   -5 │   60 │   20 │  208 │  227 │  210 │
├──────┼──────┼──────┼──────┼──────┼──────┤
│   15 │   10 │   20 │  213 │  205 │  220 │
├──────┼──────┼──────┼──────┼──────┼──────┤
│   15 │   60 │   10 │  220 │  213 │  215 │
╘══════╧══════╧══════╧══════╧══════╧══════╛
Regression equation: y = 219.117 + -0.083x1 + 0.013x2 + -0.267x3

Cochran test:
Dispersions:
d1 = 108.667
d2 = 72.667
d3 = 37.556
d4 = 8.667

gp = 0.478
✅Cochran’s C test passed

Student's t-test
Sb = 2.177
Beta:
b0 = 215.167
b1 = -0.833
b2 = 0.333
b3 = -1.333

t:
t0 = 98.822
t1 = 0.383
t2 = 0.153
t3 = 0.612

t1,t2,t3 < t_tabl(t_tabl=2.306)
Factors b2,b3,b4 can be excluded
Regression equation without excluded factors:
y = 219.117 

y1 = 219.117
y2 = 219.117
y3 = 219.117
y4 = 219.117

F-test
d = 3
S2ad = 218.230
Fp = 3.836
✅F-test passed/model is adequate
```
