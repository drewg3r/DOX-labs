"""
 * Copyright © 2021 drewg3r
 * https://github.com/drewg3r/DOX-labs

main.py: main file to run the program.
"""

import random
from tabulate import tabulate

x_max = 20 # generate x1, x2, x3 in 0...x_max range

a0 = 2
a1 = 7
a2 = 5
a3 = 10

table_headers = ["X1", "X2", "X3", "Y", "X1n", "X2n", "X3n"]

x1 = []
x2 = []
x3 = []

y  = []

x1n = []
x2n = []
x3n = []

table = []

# Generate random x1, x2, x3 and calculate y
for i in range(8):
	x1.append(random.randint(0, x_max))
	x2.append(random.randint(0, x_max))
	x3.append(random.randint(0, x_max))
	y.append(a0 + a1*x1[-1] + a2*x2[-1] + a3*x3[-1])

# Calculate x0 and dx for all factors
x10 = (max(x1)+min(x1))/2
x20 = (max(x2)+min(x2))/2
x30 = (max(x3)+min(x3))/2

dx1 = x10-min(x1)
dx2 = x20-min(x2)
dx3 = x30-min(x3)

# Calculate normalized factor's values
for i in range(8):
	x1n.append((x1[i] - x10)/dx1)
	x2n.append((x2[i] - x20)/dx2)
	x3n.append((x3[i] - x30)/dx3)
	# Generates table for factors and their normalized values
	table.append([x1[i], x2[i], x3[i], y[i], x1n[-1], x3n[-1], x3n[-1]])

# Generate table for x0 and dx
table2 = []
table2.append(["", "x1", "x2", "x3"])
table2.append(["x0", x10, x20, x30])
table2.append(["dx", dx1, dx2, dx3])

# Printing tables
print("Factors table")
print(tabulate(table, headers=table_headers, floatfmt=".4f",
	  tablefmt='fancy_grid', colalign="center"))

print("\nx0 and dx")
print(tabulate(table2, floatfmt=".4f", tablefmt='fancy_grid'))

# Finding the lowest y greater than average
y_avg = sum(y)/len(y)
y_min_greater_avg = min([x for x in y if x > y_avg])
y_index = y.index(y_min_greater_avg) # x1, x2, x3 index corresponding to found y

print("\nAverage Y: {}; the lowest Y greater than average: {}\n"\
	  "Corresponding values: x1={}, x2={}, x3={}"
	  .format(y_avg, y_min_greater_avg, x1[y_index], x2[y_index], x3[y_index]))
