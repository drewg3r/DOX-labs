# Lab2: Conducting a two-factor experiment with using linear regression

**Objective:** To conduct a two-factor experiment, to check the homogeneity of the variance according to Romanovsky's criterion, to obtain the coefficients of the regression equation, to carry out the naturalization of the regression equation. 

## Task
To check the uniformity of dispersion and get naturalized regression equation

## Test run
```
> python3 ./main.py 
DOX Lab2
╒══════╤══════╤══════╤══════╤══════╤══════╤══════╕
│   X1 │   X2 │   Y1 │   Y2 │   Y3 │   Y4 │   Y5 │
╞══════╪══════╪══════╪══════╪══════╪══════╪══════╡
│   -1 │   -1 │   26 │  -20 │   46 │   13 │  -14 │
├──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│    1 │   -1 │   61 │   69 │    8 │   45 │   62 │
├──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│   -1 │    1 │   30 │    5 │   16 │   22 │   10 │
╘══════╧══════╧══════╧══════╧══════╧══════╧══════╛

Average Y1: 10.200
Average Y2: 49.000
Average Y3: 16.600

DISPERSIONS:
d1 = 607.360
d2 = 482.000
d3 = 77.440

Main deviation: 1.789

Fuv1 = 1.260
Fuv2 = 0.128
Fuv3 = 0.161

sigma_uv1 = 0.756
sigma_uv2 = 0.077
sigma_uv3 = 0.096

Ruv1 = 0.136
Ruv2 = 0.516
Ruv3 = 0.505

Normalized regression equation: y = 32.800 + 19.400x1 + 3.200x2
Naturalized regression equation: y = 18.620 + 1.940x1 + 0.128x2
```