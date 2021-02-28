# Lab1: general principles of organization of experiments with arbitary factor's values

**Objective:** To study the basic concepts, definitions, principles of the theory of experiment planning, on the basis of which to study the construction of formalized algorithms for conducting an experiment and obtaining a formalized model of the object. Consolidate the acquired knowledge by their practical use in writing a program that implements tasks for laboratory work. 

## Task
As assigned by laboratory work, we must:
1. generate 3*8 random numbers in specified range(factors),
2. calculate Y by linear regression formula using specified a0, a1, a2, a3 params,
3. normalize factors,
4. find the lowest Y greater than average.

## Test run
```
> python3 ./main.py 
Factors table
╒══════╤══════╤══════╤═════╤═════════╤═════════╤═════════╕
│   X1 │   X2 │   X3 │   Y │     X1n │     X2n │     X3n │
╞══════╪══════╪══════╪═════╪═════════╪═════════╪═════════╡
│ 20   │ 15   │ 19   │ 407 │ 1.0000  │ 0.8571  │  0.8571 │
├──────┼──────┼──────┼─────┼─────────┼─────────┼─────────┤
│ 18   │ 6    │ 6    │ 218 │ 0.7778  │ -1.0000 │ -1.0000 │
├──────┼──────┼──────┼─────┼─────────┼─────────┼─────────┤
│ 9    │ 1    │ 7    │ 140 │ -0.2222 │ -0.8571 │ -0.8571 │
├──────┼──────┼──────┼─────┼─────────┼─────────┼─────────┤
│ 2    │ 10   │ 20   │ 266 │ -1.0000 │ 1.0000  │  1.0000 │
├──────┼──────┼──────┼─────┼─────────┼─────────┼─────────┤
│ 4    │ 3    │ 8    │ 125 │ -0.7778 │ -0.7143 │ -0.7143 │
├──────┼──────┼──────┼─────┼─────────┼─────────┼─────────┤
│ 13   │ 20   │ 18   │ 373 │ 0.2222  │ 0.7143  │  0.7143 │
├──────┼──────┼──────┼─────┼─────────┼─────────┼─────────┤
│ 10   │ 5    │ 16   │ 257 │ -0.1111 │ 0.4286  │  0.4286 │
├──────┼──────┼──────┼─────┼─────────┼─────────┼─────────┤
│ 12   │ 16   │ 18   │ 346 │ 0.1111  │ 0.7143  │  0.7143 │
╘══════╧══════╧══════╧═════╧═════════╧═════════╧═════════╛

x0 and dx
╒════╤══════╤══════╤══════╕
│    │ x1   │ x2   │ x3   │
├────┼──────┼──────┼──────┤
│ x0 │ 11.0 │ 10.5 │ 13.0 │
├────┼──────┼──────┼──────┤
│ dx │ 9.0  │ 9.5  │ 7.0  │
╘════╧══════╧══════╧══════╛

Average Y: 266.5; the lowest Y greater than average: 346
Corresponding values: x1=12, x2=16, x3=18
```